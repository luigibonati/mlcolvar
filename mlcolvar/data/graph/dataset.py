import torch
import torch_geometric as tg
import numpy as np
from typing import List, Union

from mlcolvar.data.graph import atomic
from mlcolvar.data.graph.neighborhood import get_neighborhood
# from mlcolvar.data.graph.utils import torch_tools # moved here one hot
from mlcolvar.utils.plot import pbar # moved

from mlcolvar.data.dataset import DictDataset

"""
Build the graph data from a configuration. This module is taken from MACE:
https://github.com/ACEsuit/mace/blob/main/mace/data/atomic_data.py
"""

__all__ = [
    'GraphDataSet',
    'create_dataset_from_configurations',
    'save_dataset',
    'save_dataset_as_exyz',
    'load_dataset'
]


class GraphDataSet(list):
    """
    A very simple graph dataset class.

    Parameters
    ----------
    data: List[torch_geometric.data.Data]
        The data.
    atomic_numbers: List[int]
        The atomic numbers used to build the node attributes.
    cutoff: float
        The graph cutoff radius.
    """

    def __init__(
        self,
        data: List[tg.data.Data],
        atomic_numbers: List[int],
        cutoff: float
    ) -> None:
        super().__init__()
        self.extend(data)
        self.__atomic_numbers = list(atomic_numbers)
        self.__cutoff = cutoff

    def __getitem__(
        self,
        index: Union[int, slice, list, range, np.ndarray]
    ) -> Union['GraphDataSet', tg.data.Data]:
        """
        Build sub-dataset from the dataset.

        Parameters
        ----------
        index : int, slice or list
            Indices of the data.
        """
        if type(index) in [slice, list, np.ndarray, range]:
            if isinstance(index, slice):
                index = list(range(len(self)))[index]
            data = [super(GraphDataSet, self).__getitem__(i) for i in index]
            return GraphDataSet(data, self.atomic_numbers, self.cutoff)
        elif np.issubdtype(type(index), np.integer):
            return super(GraphDataSet, self).__getitem__(index)
        else:
            raise RuntimeError(
                'Could only indexing a GraphDataSet by an int, slice or list!'
            )

    def __repr__(self) -> str:
        result = 'GRAPHDATASET [ '

        data_string = '\033[32m{:d}\033[0m\033[36m 󰡷 \033[0m'
        result = result + data_string.format(len(self))
        result = result + '| '
        data_string = '[\033[32m{}\033[0m]\033[36m 󰝨 \033[0m'
        result = result + data_string.format(
            ('{:d} ' * len(self.atomic_numbers)).strip()
        ).format(*self.atomic_numbers)
        result = result + '| '
        data_string = '\033[32m{:f}\033[0m\033[36m 󰳁 \033[0m'
        result = result + data_string.format(self.cutoff)
        result = result + ']'

        return result

    @property
    def cutoff(self) -> float:
        """
        The graph cutoff radius.
        """
        return self.__cutoff

    @property
    def atomic_numbers(self) -> List[int]:
        """
        The atomic numbers used to build the node attributes.
        """
        return self.__atomic_numbers.copy()


def _create_dataset_from_configuration(
    config: atomic.Configuration,
    z_table: atomic.AtomicNumberTable,
    cutoff: float,
    buffer: float = 0.0
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
    buffer: float
        Buffer size used in finding active environment atoms.
    """

    assert config.graph_labels is None or len(config.graph_labels.shape) == 2

    # NOTE: here we do not take care about the nodes that are not taking part
    # the graph, like, we don't even change the node indices in `edge_index`.
    # Here we simply ignore them, and rely on the `RemoveIsolatedNodes` method
    # that will be called later (in `create_dataset_from_configurations`).
    edge_index, shifts, unit_shifts = get_neighborhood(
        positions=config.positions,
        cutoff=cutoff,
        cell=config.cell,
        pbc=config.pbc,
        system_indices=config.system,
        environment_indices=config.environment,
        buffer=buffer
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
    one_hot = to_one_hot(
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

    n_system = (
        torch.tensor(
            [[len(config.system)]], dtype=torch.get_default_dtype()
        ) if config.system is not None
        else torch.tensor(
            [[one_hot.shape[0]]], dtype=torch.get_default_dtype()
        )
    )

    if config.system is not None:
        system_masks = torch.zeros((one_hot.shape[0], 1), dtype=torch.bool)
        system_masks[config.system, 0] = 1
    else:
        system_masks = None

    return tg.data.Data(
        edge_index=edge_index,
        shifts=shifts,
        unit_shifts=unit_shifts,
        positions=positions,
        cell=cell,
        node_attrs=one_hot,
        node_labels=node_labels,
        graph_labels=graph_labels,
        n_system=n_system,
        system_masks=system_masks,
        weight=weight,
    )


def create_dataset_from_configurations(
    config: atomic.Configurations,
    z_table: atomic.AtomicNumberTable,
    cutoff: float,
    buffer: float = 0.0,
    remove_isolated_nodes: bool = False,
    show_progress: bool = True
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
    buffer: float
        Buffer size used in finding active environment atoms.
    remove_isolated_nodes: bool
        If remove isolated nodes from the dataset.
    show_progress: bool
        If show the progress bar.
    """
    if show_progress:
        items = pbar(config, frequency=0.0001, prefix='Making graphs')
    else:
        items = config

    data_list = [
        _create_dataset_from_configuration(
            c, z_table, cutoff, buffer
        ) for c in items
    ]

    if remove_isolated_nodes:
        # TODO: not the worst way to fake the `is_node_attr` method of
        # `tg.data.storage.GlobalStorage` ...
        # I mean, when there are exact three atoms in the graph, the
        # `RemoveIsolatedNodes` method will remove the cell vectors that
        # correspond to the isolated node ... This is a consequence of that
        # pyg regarding the cell vectors as some kind of node features.
        # So here we first remove the isolated nodes, then set the cell back.
        cell_list = [d.cell.clone() for d in data_list]
        transform = tg.transforms.remove_isolated_nodes.RemoveIsolatedNodes()
        data_list = [transform(d) for d in data_list]
        for i in range(len(data_list)):
            data_list[i].cell = cell_list[i]

    # dataset = GraphDataSet(data_list, z_table.zs, cutoff)
    dataset = DictDataset(dictionary={'data_list' : data_list},
                          metadata={'z_table' : z_table.zs,
                                    'cutoff' : cutoff})

    return dataset


def save_dataset(dataset: GraphDataSet, file_name: str) -> None:
    """
    Save a dataset to disk.

    Parameters
    ----------
    dataset: GraphDataSet
        The dataset.
    file_name: str
        The filename.
    """
    assert isinstance(dataset, GraphDataSet)

    torch.save(dataset, file_name)  # super torch magic go brrrrrrrrr


def load_dataset(file_name: str) -> GraphDataSet:
    """
    Load a dataset from disk.

    Parameters
    ----------
    file_name: str
        The filename.
    """
    dataset = torch.load(file_name)

    assert isinstance(dataset, GraphDataSet)

    return dataset


def save_dataset_as_exyz(dataset: GraphDataSet, file_name: str) -> None:
    """
    Save a dataset to disk in the extxyz format.

    Parameters
    ----------
    dataset: GraphDataSet
        The dataset.
    file_name: str
        The filename.
    """
    z_table = atomic.AtomicNumberTable.from_zs(dataset.atomic_numbers)

    fp = open(file_name, 'w')

    for d in dataset:
        print(len(d['positions']), file=fp)
        line = (
            'Lattice="{:s}" '.format((r'{:.5f} ' * 9).strip())
            + 'Properties=species:S:1:pos:R:3 pbc="T T T"'
        )
        cell = [c.item() for c in d['cell'].flatten()]
        print(line.format(*cell), file=fp)
        for i in range(0, len(d['positions'])):
            s = z_table.index_to_symbol(np.where(d['node_attrs'][i])[0][0])
            print('{:2s}'.format(s), file=fp, end=' ')
            positions = [p.item() for p in d['positions'][i]]
            print('{:10.5f} {:10.5f} {:10.5f}'.format(*positions), file=fp)

    fp.close()

def to_one_hot(indices: torch.Tensor, n_classes: int) -> torch.Tensor:
    """
    Generates one-hot encoding with `n_classes` classes from `indices`

    Parameters
    ----------
    indices: torch.Tensor (shape: [N, 1])
        Node incices.
    n_classes: int
        Number of classes.

    Returns
    -------
    encoding: torch.tensor (shape: [N, n_classes])
        The one-hot encoding.
    """
    shape = indices.shape[:-1] + (n_classes,)
    oh = torch.zeros(shape, device=indices.device).view(shape)

    # scatter_ is the in-place version of scatter
    oh.scatter_(dim=-1, index=indices, value=1)

    return oh.view(*shape)


def test_from_configuration() -> None:
    numbers = [8, 1, 1]
    positions = np.array(
        [[0.0, 0.0, 0.0], [0.07, 0.07, 0.0], [0.07, -0.07, 0.0]],
        dtype=float
    )
    cell = np.identity(3, dtype=float) * 0.2
    graph_labels = np.array([[1]])
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
    data = _create_dataset_from_configuration(config, z_table, 0.1)
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
    assert (data['graph_labels'] == torch.tensor([[1.0]])).all()
    assert data['weight'] == 1.0

    config = atomic.Configuration(
        atomic_numbers=numbers,
        positions=positions,
        cell=cell,
        pbc=[True] * 3,
        node_labels=node_labels,
        graph_labels=graph_labels,
        system=[1],
        environment=[2]
    )
    data = _create_dataset_from_configuration(config, z_table, 0.1)
    assert (
        data['edge_index'] == torch.tensor([[1, 2], [2, 1]])
    ).all()
    assert (
        data['shifts'] == torch.tensor([[0.0, 0.2, 0.0], [0.0, -0.2, 0.0]])
    ).all()
    assert (
        data['unit_shifts'] == torch.tensor(
            [[0.0, 1.0, 0.0], [0.0, -1.0, 0.0]]
        )
    ).all()

    config = atomic.Configuration(
        atomic_numbers=numbers,
        positions=positions,
        cell=cell,
        pbc=[True] * 3,
        node_labels=node_labels,
        graph_labels=graph_labels,
        system=[0],
        environment=[1, 2]
    )
    data = _create_dataset_from_configuration(config, z_table, 0.1)
    assert (
        data['edge_index'] == torch.tensor(
            [[0, 0, 1, 1, 2, 2], [2, 1, 0, 2, 1, 0]]
        )
    ).all()

    positions = np.array(
        [[0.0, 0.0, 0.0], [0.07, 0.07, 0.0], [0.07, -0.08, 0.0]],
        dtype=float
    )
    config = atomic.Configuration(
        atomic_numbers=numbers,
        positions=positions,
        cell=cell,
        pbc=[True] * 3,
        node_labels=node_labels,
        graph_labels=graph_labels,
        system=[0],
        environment=[1, 2]
    )
    data = _create_dataset_from_configuration(config, z_table, 0.1)
    assert (
        data['edge_index'] == torch.tensor([[0, 1], [1, 0]])
    ).all()
    data = _create_dataset_from_configuration(config, z_table, 0.11)
    assert (
        data['edge_index'] == torch.tensor(
            [[0, 0, 1, 1, 2, 2], [2, 1, 0, 2, 1, 0]]
        )
    ).all()
    data = _create_dataset_from_configuration(config, z_table, 0.1, 0.01)
    assert (
        data['edge_index'] == torch.tensor(
            [[0, 1, 1, 2], [1, 0, 2, 1]]
        )
    ).all()
    assert (
        data['shifts'] == torch.tensor([
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.2, 0.0],
            [0.0, -0.2, 0.0]
        ])
    ).all()
    assert (
        data['unit_shifts'] == torch.tensor([
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0]
        ])
    ).all()

    config = [atomic.Configuration(
        atomic_numbers=numbers,
        positions=positions,
        cell=cell,
        pbc=[True] * 3,
        node_labels=node_labels,
        graph_labels=np.array([[i]]),
    ) for i in range(0, 10)]
    dataset = create_dataset_from_configurations(
        config, z_table, 0.1, show_progress=False
    )

    dataset_1 = dataset[range(0, 5, 2)]
    assert dataset_1.atomic_numbers == [1, 8]
    assert (dataset_1[0]['graph_labels'] == torch.tensor([[0.0]])).all()
    assert (dataset_1[1]['graph_labels'] == torch.tensor([[2.0]])).all()
    assert (dataset_1[2]['graph_labels'] == torch.tensor([[4.0]])).all()

    dataset_1 = dataset[np.array([0, -1])]
    assert dataset_1.atomic_numbers == [1, 8]
    assert (dataset_1[0]['graph_labels'] == torch.tensor([[0.0]])).all()
    assert (dataset_1[1]['graph_labels'] == torch.tensor([[9.0]])).all()


def test_from_configurations() -> None:
    numbers = [8, 1, 1]
    positions = np.array(
        [[0.0, 0.0, 0.0], [0.07, 0.07, 0.0], [0.07, -0.07, 0.0]],
        dtype=float
    )
    cell = np.identity(3, dtype=float) * 0.2
    graph_labels = np.array([[1]])
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
    data = create_dataset_from_configurations(
        [config], z_table, 0.1, remove_isolated_nodes=True, show_progress=False
    )[0]
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
    assert (data['graph_labels'] == torch.tensor([[1.0]])).all()
    assert data['weight'] == 1.0

    config = atomic.Configuration(
        atomic_numbers=numbers,
        positions=positions,
        cell=cell,
        pbc=[True] * 3,
        node_labels=node_labels,
        graph_labels=graph_labels,
        system=[1],
        environment=[2]
    )
    data = create_dataset_from_configurations(
        [config], z_table, 0.1, remove_isolated_nodes=True, show_progress=False
    )[0]
    assert (
        data['positions'] == torch.tensor([
            [0.07, 0.07, 0.0], [0.07, -0.07, 0.0],
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
            [1.0, 0.0], [1.0, 0.0]
        ])
    ).all()
    assert (
        data['edge_index'] == torch.tensor([[0, 1], [1, 0]])
    ).all()
    assert (
        data['shifts'] == torch.tensor([[0.0, 0.2, 0.0], [0.0, -0.2, 0.0]])
    ).all()
    assert (
        data['unit_shifts'] == torch.tensor(
            [[0.0, 1.0, 0.0], [0.0, -1.0, 0.0]]
        )
    ).all()


if __name__ == '__main__':
    test_from_configuration()
    test_from_configurations()
