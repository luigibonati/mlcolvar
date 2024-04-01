import torch
import torch_geometric as tg
import numpy as np

import mlcolvar.graph.utils as gutils

from neighborhood import get_neighborhood

"""
The graph data class. This module is taken from MACE:
https://github.com/ACEsuit/mace/blob/main/mace/data/atomic_data.py
"""

__all__ = ['GraphData']


class GraphData(tg.data.Data):
    """
    The graph data class.

    Parameters
    ----------
    edge_index: torch.Tensor (shape: [2, n_edges])
        The edge index array.
    node_attrs: torch.Tensor (shape: [n_nodes, n_node_attrs])
        The node attribute array.
    positions: torch.Tensor (shape: [n_nodes, 3], units: nm)
        The node positions.
    shifts: torch.Tensor (shape: [n_edges, 3], units: nm)
        The shift array.
    unit_shifts: torch.Tensor (shape: [n_edges, 3])
        The unit shift array.
    cell: torch.Tensor (shape: [3, 3], units: nm)
        The lattice vectors.
    node_labels: torch.Tensor (shape: [n_nodes, n_node_labels])
        The node labels.
    graph_labels: torch.Tensor (shape: [n_graph_labels])
        The graph labels.
    weight: torch.Tensor (shape: [])
        The configuration weight in the loss funciton.
    """
    batch: torch.Tensor
    edge_index: torch.Tensor
    node_attrs: torch.Tensor
    edge_vectors: torch.Tensor
    edge_lengths: torch.Tensor
    positions: torch.Tensor
    shifts: torch.Tensor
    unit_shifts: torch.Tensor
    cell: torch.Tensor
    node_labels: torch.Tensor
    graph_labels: torch.Tensor
    weight: torch.Tensor

    def __init__(
        self,
        edge_index: torch.Tensor,
        node_attrs: torch.Tensor,
        positions: torch.Tensor,
        shifts: torch.Tensor,
        unit_shifts: torch.Tensor,
        cell: torch.Tensor,
        node_labels: torch.Tensor,
        graph_labels: torch.Tensor,
        weight: torch.Tensor,
    ):
        # Check shapes
        num_nodes = node_attrs.shape[0]
        assert edge_index.shape[0] == 2 and len(edge_index.shape) == 2
        assert positions.shape == (num_nodes, 3)
        assert shifts.shape[1] == 3
        assert unit_shifts.shape[1] == 3
        assert cell is None or cell.shape == (3, 3)

        # Aggregate data
        data = {
            'num_nodes': num_nodes,
            'edge_index': edge_index,
            'shifts': shifts,
            'unit_shifts': unit_shifts,
            'positions': positions,
            'cell': cell,
            'node_attrs': node_attrs,
            'node_labels': node_labels,
            'graph_labels': graph_labels,
            'weight': weight,
        }
        super().__init__(**data)

    @classmethod
    def from_config(
        cls,
        config: gutils.atomic.Configuration,
        z_table: gutils.atomic.AtomicNumberTable,
        cutoff: float,
    ) -> 'GraphData':
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
        # fmt: off
        assert (
            (config.node_labels is not None) or
            (config.graph_labels is not None)
        ), "Either node labels or graph labels should be given!"
        # fmt: on

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
        one_hot = gutils.torch_tools.to_one_hot(
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

        return cls(
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


def test_graph_data():
    numbers = [8, 1, 1]
    positions = np.array(
        [[0.0, 0.0, 0.0], [0.07, 0.07, 0.0], [0.07, -0.07, 0.0]],
        dtype=float
    )
    cell = np.identity(3, dtype=float) * 0.2
    graph_labels = np.array([1])
    node_labels = np.array([[0], [1], [1]])
    z_table = gutils.atomic.AtomicNumberTable.from_zs(numbers)

    config = gutils.atomic.Configuration(
        atomic_numbers=numbers,
        positions=positions,
        cell=cell,
        pbc=[True] * 3,
        node_labels=node_labels,
        graph_labels=graph_labels,
    )
    data = GraphData.from_config(config, z_table, 0.1)

    assert data['num_nodes'] == 3

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

    config = gutils.atomic.Configuration(
        atomic_numbers=numbers,
        positions=positions,
        cell=cell,
        pbc=[True] * 3,
        node_labels=node_labels,
        graph_labels=graph_labels,
        edge_senders=[0],
    )
    data = GraphData.from_config(config, z_table, 0.1)

    assert (
        data['edge_index'] == torch.tensor([[0, 0], [2, 1]])
    ).all()

    config = gutils.atomic.Configuration(
        atomic_numbers=numbers,
        positions=positions,
        cell=cell,
        pbc=[True] * 3,
        node_labels=node_labels,
        graph_labels=graph_labels,
        edge_receivers=[1, 2],
    )
    data = GraphData.from_config(config, z_table, 0.1)

    assert (
        data['edge_index'] == torch.tensor([[0, 0, 1, 2], [2, 1, 2, 1]])
    ).all()

    config = gutils.atomic.Configuration(
        atomic_numbers=numbers,
        positions=positions,
        cell=cell,
        pbc=[True] * 3,
        node_labels=node_labels,
        graph_labels=graph_labels,
        edge_senders=[1, 2],
        edge_receivers=[1, 2],
    )
    data = GraphData.from_config(config, z_table, 0.1)

    assert (data['edge_index'] == torch.tensor([[1, 2], [2, 1]])).all()


if __name__ == '__main__':
    test_graph_data()
