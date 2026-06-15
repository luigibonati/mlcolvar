import torch
import numpy as np
from typing import List   

from mlcolvar.data import DictDataset
from mlcolvar.data.graph.atomic import AtomicNumberTable

__all__ = ["save_dataset", "load_dataset", "save_dataset_configurations_as_extyz"]

def save_dataset(dataset: DictDataset, file_name: str) -> None:
    """Save a dataset to disk.

    Parameters
    ----------
    dataset: DictDataset
        Dataset to be saved
    file_name: str
        Name of the file to save to
    """
    assert isinstance(dataset, DictDataset)

    torch.save(dataset, file_name)


def load_dataset(file_name: str) -> DictDataset:
    """Load a dataset from disk.

    Parameters
    ----------
    file_name: str
        Name of the file to load the dataset from
    """
    dataset = torch.load(file_name, weights_only=False)

    assert isinstance(dataset, DictDataset)

    return dataset


def save_dataset_configurations_as_extyz(dataset: DictDataset, 
                                         file_name: str,
                                         extra_keys: List[str] = None) -> None:
    """Save a dataset to disk in the extxyz format.

    Parameters
    ----------
    dataset: DictDataset
        Dataset to be saved with data_type graphs
    file_name: str
        Name of the file to save to
    extra_keys: list[str]
        Extra keys of data_list to be printed on the extxyz. 
        Only single per-atom quantities are allowed.
    """
    # check the dataset type is 'graphs'
    if not dataset.metadata["data_type"] == "graphs":
        raise(
            ValueError("Can only save to extxyz dataset with data_type='graphs'!")
        )
    
    extra_keys_string = ''
    if extra_keys is not None:
        for key in extra_keys:
            extra_key_value = dataset['data_list'][0][key]
            if extra_key_value.shape[0] != len(dataset['data_list'][0]['positions']) or extra_key_value.shape[-1] != 1:
                raise ValueError(f"The selected extra_key {key} is not a single per-atom quantity!")
            extra_keys_string = extra_keys_string + f':{key}:R:{extra_key_value.shape[-1]}'
    else:
        extra_keys = []
    
    # initialize the atomic number object
    atomic_numbers = dataset.metadata.get("atomic_numbers", None)
    if atomic_numbers is None:
        raise KeyError("Dataset metadata missing 'atomic_numbers'.")
    atomic_numbers = AtomicNumberTable.from_zs(atomic_numbers)

    # create file
    fp = open(file_name, 'w')

    for i in range(len(dataset)):
        d = dataset[i]['data_list']

        # print number of atoms
        print(len(d['positions']), file=fp)

        # header line for configuration d
        # Lattice, properties, pbc
        line = (
            'Lattice="{:s}" '.format((r'{:.5f} ' * 9).strip())
            + 'Properties=species:S:1:pos:R:3'+
            extra_keys_string +
            ' pbc="T T T"'
        )

        # cell info
        cell = [c.item() for c in d['cell'].flatten()]
        print(line.format(*cell), file=fp)

        # write atoms positions
        for j in range(0, len(d['positions'])):
            # chemical symbol
            s = atomic_numbers.index_to_symbol(np.where(d['node_attrs'][j])[0][0])
            print('{:2s}'.format(s), file=fp, end='')

            # positions
            positions = [p.item() for p in d['positions'][j]]
            print(' {:10.5f} {:10.5f} {:10.5f}'.format(*positions), file=fp, end='')
            for key in extra_keys:
                print(' {:10.5f}'.format(d[key][j][0]), file=fp, end='')
            print('', file=fp)
            
    fp.close()




import tempfile

def test_save_dataset():
    # check using descriptors dataset
    dataset_dict = {
        "data": torch.Tensor([[1.0], [2.0], [0.3], [0.4]]),
        "labels": [0, 0, 1, 1],
        "weights": np.asarray([0.5, 1.5, 1.5, 0.5]),
    }
    dataset = DictDataset(dataset_dict)

    # save to temporary working directory
    with tempfile.TemporaryDirectory() as tmpdir:
        save_dataset(dataset=dataset, file_name=f'{tmpdir}/saved_dataset')

        # load and check it's ok
        loaded = load_dataset(file_name=f'{tmpdir}/saved_dataset') 
        assert(torch.allclose(dataset['data'], loaded['data'])) 

    # check using graph dataset
    from mlcolvar.data.graph.atomic import AtomicNumberTable, Configuration
    from mlcolvar.data.graph.utils import create_dataset_from_configurations
    numbers = [8, 1, 1]
    positions = np.array(
        [[0.0, 0.0, 0.0], [0.07, 0.07, 0.0], [0.07, -0.07, 0.0]],
        dtype=float
    )
    cell = np.identity(3, dtype=float) * 0.2
    graph_labels = np.array([[1]])
    node_labels = np.array([[0], [1], [1]])
    atomic_numbers = AtomicNumberTable.from_zs(numbers)

    config = [Configuration(
        atomic_numbers=numbers,
        positions=positions,
        cell=cell,
        pbc=[True] * 3,
        node_labels=node_labels,
        graph_labels=graph_labels,
    )]
    dataset = create_dataset_from_configurations(
                config, atomic_numbers, 0.1, show_progress=False
                )

    # save dataset
    with tempfile.TemporaryDirectory() as tmpdir:
        save_dataset(dataset=dataset, file_name=f'{tmpdir}/saved_dataset')

        # load and check it's ok
        loaded = load_dataset(file_name=f'{tmpdir}/saved_dataset') 
        assert(torch.allclose(dataset['data_list'][0]['positions'], loaded['data_list'][0]['positions'])) 

    # save to extxyz
    with tempfile.TemporaryDirectory() as tmpdir:
        save_dataset_configurations_as_extyz(dataset=dataset, file_name=f'{tmpdir}/saved_dataset')
        
