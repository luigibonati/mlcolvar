import tempfile

import numpy as np
import torch

from mlcolvar.data import DictDataset
from mlcolvar.data.graph.atomic import AtomicNumberTable, Configuration
from mlcolvar.data.utils import (
    load_dataset,
    save_dataset,
    save_dataset_configurations_as_extyz,
)


def test_save_dataset():
    # Descriptor dataset
    dataset_dict = {
        "data": torch.tensor([[1.0], [2.0], [0.3], [0.4]]),
        "labels": [0, 0, 1, 1],
        "weights": np.asarray([0.5, 1.5, 1.5, 0.5]),
    }
    dataset = DictDataset(dataset_dict)

    with tempfile.TemporaryDirectory() as tmpdir:
        file_name = f"{tmpdir}/saved_dataset"

        save_dataset(
            dataset=dataset,
            file_name=file_name,
        )

        loaded = load_dataset(
            file_name=file_name,
        )

        assert torch.allclose(
            dataset["data"],
            loaded["data"],
        )

    # Graph dataset
    numbers = [8, 1, 1]
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.07, 0.07, 0.0],
            [0.07, -0.07, 0.0],
        ],
        dtype=float,
    )
    cell = np.identity(3, dtype=float) * 0.2
    graph_labels = np.array([[1]])
    node_labels = np.array([[0], [1], [1]])

    atomic_numbers = AtomicNumberTable.from_zs(numbers)

    config = [
        Configuration(
            atomic_numbers=numbers,
            positions=positions,
            cell=cell,
            pbc=[True] * 3,
            node_labels=node_labels,
            graph_labels=graph_labels,
        )
    ]

    dataset = DictDataset.graph_from_configurations(
        config=config,
        atomic_numbers=atomic_numbers,
        cutoff=0.1,
        show_progress=False,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        file_name = f"{tmpdir}/saved_dataset"

        save_dataset(
            dataset=dataset,
            file_name=file_name,
        )

        loaded = load_dataset(
            file_name=file_name,
        )

        assert torch.allclose(
            dataset["data_list"][0]["positions"],
            loaded["data_list"][0]["positions"],
        )

    # Save graph dataset to extxyz
    with tempfile.TemporaryDirectory() as tmpdir:
        save_dataset_configurations_as_extyz(
            dataset=dataset,
            file_name=f"{tmpdir}/saved_dataset",
        )