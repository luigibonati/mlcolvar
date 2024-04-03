import torch
import torch_geometric as tg
import numpy as np
from typing import List

from mlcolvar.graph.data import atomic
from mlcolvar.graph.data.neighborhood import get_neighborhood
from mlcolvar.graph.utils import torch_tools

"""
Build the graph data from a configuration. This module is taken from MACE:
https://github.com/ACEsuit/mace/blob/main/mace/data/atomic_data.py
"""

__all__ = ['GraphDataSet', 'from_configuration', 'from_configurations']


GraphDataSet = List[tg.data.Data]


def from_configuration(
    config: atomic.Configuration,
    z_table: atomic.AtomicNumberTable,
    cutoff: float,
) -> tg.data.Data:
    """
    Build the graph data object from a configuration.

    Parameters
    ----------
    config: mlcolvar.graph.utils.atomic.Configuration
        The configuration.
    z_table: mlcolvar.graph.utils.atomic.AtomicNumberTable
        The atomic number table used to build the node attributes.
    cutoff: float
        The graph cutoff radius.
    """
    edge_index, shifts, unit_shifts = get_neighborhood(
        positions=config.positions,
        cutoff=cutoff,
        cell=config.cell,
        pbc=config.pbc,
        sender_indices=config.edge_senders,
        receiver_indices=config.edge_receivers,
    )
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    shifts = torch.tensor(shifts, dtype=torch.get_default_dtype())
    unit_shifts = torch.tensor(
        unit_shifts, dtype=torch.get_default_dtype()
    )

    positions = torch.tensor(
        config.positions, dtype=torch.get_default_dtype()
    )
    cell = torch.tensor(config.cell, dtype=torch.get_default_dtype())

    indices = z_table.zs_to_indices(config.atomic_numbers)
    one_hot = torch_tools.to_one_hot(
        torch.tensor(indices, dtype=torch.long).unsqueeze(-1),
        n_classes=len(z_table),
    )

    node_labels = (
        torch.tensor(config.node_labels, dtype=torch.get_default_dtype())
        if config.node_labels is not None
        else None
    )

    graph_labels = (
        torch.tensor(config.graph_labels, dtype=torch.get_default_dtype())
        if config.graph_labels is not None
        else None
    )

    weight = (
        torch.tensor(config.weight, dtype=torch.get_default_dtype())
        if config.weight is not None
        else 1
    )

    return tg.data.Data(
        edge_index=edge_index,
        shifts=shifts,
        unit_shifts=unit_shifts,
        positions=positions,
        cell=cell,
        node_attrs=one_hot,
        node_labels=node_labels,
        graph_labels=graph_labels,
        weight=weight,
    )


def from_configurations(
    config: atomic.Configurations,
    z_table: atomic.AtomicNumberTable,
    cutoff: float,
) -> GraphDataSet:
    """
    Build graph data objects from configurations.

    Parameters
    ----------
    config: mlcolvar.graph.utils.atomic.Configurations
        The configurations.
    z_table: mlcolvar.graph.utils.atomic.AtomicNumberTable
        The atomic number table used to build the node attributes.
    cutoff: float
        The graph cutoff radius.
    """
    return [from_configuration(c, z_table, cutoff) for c in config]


def test_from_configuration() -> None:
    numbers = [8, 1, 1]
    positions = np.array(
        [[0.0, 0.0, 0.0], [0.07, 0.07, 0.0], [0.07, -0.07, 0.0]],
        dtype=float
    )
    cell = np.identity(3, dtype=float) * 0.2
    graph_labels = np.array([1])
    node_labels = np.array([[0], [1], [1]])
    z_table = atomic.AtomicNumberTable.from_zs(numbers)

    config = atomic.Configuration(
        atomic_numbers=numbers,
        positions=positions,
        cell=cell,
        pbc=[True] * 3,
        node_labels=node_labels,
        graph_labels=graph_labels,
    )
    data = from_configuration(config, z_table, 0.1)
    assert (
        data['edge_index'] == torch.tensor(
            [[0, 0, 1, 1, 2, 2], [2, 1, 0, 2, 1, 0]]
        )
    ).all()
    assert (
        data['shifts'] == torch.tensor([
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.2, 0.0],
            [0.0, -0.2, 0.0],
            [0.0, 0.0, 0.0],
        ])
    ).all()
    assert (
        data['unit_shifts'] == torch.tensor([
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 0.0],
        ])
    ).all()
    assert (
        data['positions'] == torch.tensor([
            [0.0, 0.0, 0.0],
            [0.07, 0.07, 0.0],
            [0.07, -0.07, 0.0],
        ])
    ).all()
    assert (
        data['cell'] == torch.tensor([
            [0.2, 0.0, 0.0],
            [0.0, 0.2, 0.0],
            [0.0, 0.0, 0.2],
        ])
    ).all()
    assert (
        data['node_attrs'] == torch.tensor([
            [0.0, 1.0], [1.0, 0.0], [1.0, 0.0]
        ])
    ).all()
    assert (data['node_labels'] == torch.tensor([[0.0], [1.0], [1.0]])).all()
    assert (data['graph_labels'] == torch.tensor([1.0])).all()
    assert data['weight'] == 1.0

    config = atomic.Configuration(
        atomic_numbers=numbers,
        positions=positions,
        cell=cell,
        pbc=[True] * 3,
        node_labels=node_labels,
        graph_labels=graph_labels,
        edge_senders=[0],
    )
    data = from_configuration(config, z_table, 0.1)
    assert (
        data['edge_index'] == torch.tensor([[0, 0], [2, 1]])
    ).all()

    config = atomic.Configuration(
        atomic_numbers=numbers,
        positions=positions,
        cell=cell,
        pbc=[True] * 3,
        node_labels=node_labels,
        graph_labels=graph_labels,
        edge_receivers=[1, 2],
    )
    data = from_configuration(config, z_table, 0.1)
    assert (
        data['edge_index'] == torch.tensor([[0, 0, 1, 2], [2, 1, 2, 1]])
    ).all()

    config = atomic.Configuration(
        atomic_numbers=numbers,
        positions=positions,
        cell=cell,
        pbc=[True] * 3,
        node_labels=node_labels,
        graph_labels=graph_labels,
        edge_senders=[1, 2],
        edge_receivers=[1, 2],
    )
    data = from_configuration(config, z_table, 0.1)
    assert (data['edge_index'] == torch.tensor([[1, 2], [2, 1]])).all()


if __name__ == '__main__':
    test_from_configuration()
