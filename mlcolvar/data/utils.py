from typing import List, Optional

import torch
from ase.io import write

from mlcolvar.data import DictDataset

__all__ = [
    "save_dataset",
    "load_dataset",
    "save_dataset_configurations_as_extyz",
]


def save_dataset(dataset: DictDataset, file_name: str) -> None:
    """Save a dataset to disk.

    Parameters
    ----------
    dataset : DictDataset
        Dataset to be saved.
    file_name : str
        Name of the file to save to.
    """
    assert isinstance(dataset, DictDataset)
    torch.save(dataset, file_name)


def load_dataset(file_name: str) -> DictDataset:
    """Load a dataset from disk.

    Parameters
    ----------
    file_name : str
        Name of the file to load the dataset from.

    Returns
    -------
    DictDataset
        Loaded dataset.
    """
    dataset = torch.load(file_name, weights_only=False)
    assert isinstance(dataset, DictDataset)
    return dataset


def save_dataset_configurations_as_extyz(
    dataset: DictDataset,
    file_name: str,
    extra_keys: Optional[List[str]] = None,
) -> None:
    """Save graph configurations in extended XYZ format.

    Parameters
    ----------
    dataset : DictDataset
        Graph-based dataset to save.
    file_name : str
        Output file name.
    extra_keys : list[str], optional
        Additional scalar per-atom quantities to include.
    """
    if dataset.metadata.get("data_type") != "graphs":
        raise ValueError(
            "Can only save datasets with data_type='graphs' to extxyz."
        )

    if extra_keys is None:
        extra_keys = []

    atoms_list = dataset.to_ase()

    for atoms, graph in zip(atoms_list, dataset["data_list"]):
        for key in extra_keys:
            values = graph[key]

            if (
                values.ndim != 2
                or values.shape[0] != len(atoms)
                or values.shape[1] != 1
            ):
                raise ValueError(
                    f"The selected extra_key {key!r} is not "
                    "a scalar per-atom quantity."
                )

            atoms.new_array(
                key,
                values.detach().cpu().numpy().reshape(-1),
            )

    write(file_name, atoms_list, format="extxyz")