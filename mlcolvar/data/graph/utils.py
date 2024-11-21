import torch
import torch_geometric

from mlcolvar.data import DictDataset, DictModule
from mlcolvar.data.graph import atomic
from mlcolvar.data.graph.neighborhood import get_neighborhood
from mlcolvar.utils.plot import pbar

def _create_dataset_from_configuration(
    config: atomic.Configuration,
    z_table: atomic.AtomicNumberTable,
    cutoff: float,
    buffer: float = 0.0
) -> torch_geometric.data.Data:
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

    return torch_geometric.data.Data(
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
) -> DictDataset:
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
        # `torch_geometric.data.storage.GlobalStorage` ...
        # I mean, when there are exact three atoms in the graph, the
        # `RemoveIsolatedNodes` method will remove the cell vectors that
        # correspond to the isolated node ... This is a consequence of that
        # pyg regarding the cell vectors as some kind of node features.
        # So here we first remove the isolated nodes, then set the cell back.
        cell_list = [d.cell.clone() for d in data_list]
        transform = torch_geometric.transforms.remove_isolated_nodes.RemoveIsolatedNodes()
        data_list = [transform(d) for d in data_list]
        for i in range(len(data_list)):
            data_list[i].cell = cell_list[i]

    dataset = DictDataset(dictionary={'data_list' : data_list},
                          metadata={'z_table' : z_table.zs,
                                    'cutoff' : cutoff},
                          data_type='graphs')

    return dataset

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

def create_test_graph_input(get_example: bool = False) -> torch_geometric.data.Batch:
    numbers = [8, 1, 1]
    positions = np.array(
        [
            [[0.0, 0.0, 0.0], [0.07, 0.07, 0.0], [0.07, -0.07, 0.0]],
            [[0.0, 0.0, 0.0], [-0.07, 0.07, 0.0], [0.07, 0.07, 0.0]],
            [[0.0, 0.0, 0.0], [0.07, -0.07, 0.0], [0.07, 0.07, 0.0]],
            [[0.0, 0.0, 0.0], [0.0, -0.07, 0.07], [0.0, 0.07, 0.07]],
            [[0.0, 0.0, 0.0], [0.07, 0.0, 0.07], [-0.07, 0.0, 0.07]],
            [[0.1, 0.0, 1.1], [0.17, 0.07, 1.1], [0.17, -0.07, 1.1]],
        ],
        dtype=np.float64
    )
    cell = np.identity(3, dtype=float) * 0.2
    graph_labels = np.array([[[0]], [[1]]] * 3)
    node_labels = np.array([[0], [1], [1]])
    z_table = atomic.AtomicNumberTable.from_zs(numbers)

    config = [
        atomic.Configuration(
            atomic_numbers=numbers,
            positions=positions[i],
            cell=cell,
            pbc=[True] * 3,
            node_labels=node_labels,
            graph_labels=graph_labels[i],
        ) for i in range(0, 6)
    ]
    dataset = create_dataset_from_configurations(
        config, z_table, 0.1, show_progress=False
    )

    loader = DictModule(
        dataset,
        lengths=(1.0,),
        batch_size=10,
        shuffle=False,
    )
    loader.setup()
    if get_example:
        out = next(iter(loader.train_dataloader()))['data_list'].get_example(0)
        out['batch'] = torch.tensor([0], dtype=torch.int64)
        return out
    else:
        return next(iter(loader.train_dataloader()))


# ===============================================================================
# ===============================================================================
# ==================================== TESTS ====================================
# ===============================================================================
# ===============================================================================

import numpy as np

def test_from_configuration() -> None:
    # fake atomic numbers, positions, cell, graph label, node labels
    numbers = [8, 1, 1]
    positions = np.array([[0.0, 0.0, 0.0], 
                          [0.07, 0.07, 0.0], 
                          [0.07, -0.07, 0.0]],
                         dtype=float
                        )
    cell = np.identity(3, dtype=float) * 0.2
    graph_labels = np.array([[1]])
    node_labels = np.array([[0], [1], [1]])

    # init AtomicNumber object
    z_table = atomic.AtomicNumberTable.from_zs(numbers)

    # initialize configuration using all atoms
    config = atomic.Configuration(
        atomic_numbers=numbers,
        positions=positions,
        cell=cell,
        pbc=[True] * 3,
        node_labels=node_labels,
        graph_labels=graph_labels,
    )

    # create dataset from a configuration
    data = _create_dataset_from_configuration(config, z_table, 0.1)

    # check edges and shifts are created correctly
    assert(data['edge_index'] == torch.tensor([[0, 0, 1, 1, 2, 2],
                                                [2, 1, 0, 2, 1, 0]])
          ).all()

    assert(data['shifts'] == torch.tensor([[0.0, 0.0, 0.0],
                                           [0.0, 0.0, 0.0],
                                           [0.0, 0.0, 0.0],
                                           [0.0, 0.2, 0.0],
                                           [0.0, -0.2, 0.0],
                                           [0.0, 0.0, 0.0]])
            ).all()
    
    assert(data['unit_shifts'] == torch.tensor([[0.0, 0.0, 0.0],
                                                [0.0, 0.0, 0.0],
                                                [0.0, 0.0, 0.0],
                                                [0.0, 1.0, 0.0],
                                                [0.0, -1.0, 0.0],
                                                [0.0, 0.0, 0.0]])
            ).all()
    
    # check correct storage
    assert(data['positions'] == torch.tensor([[0.0, 0.0, 0.0],
                                              [0.07, 0.07, 0.0],
                                              [0.07, -0.07, 0.0]])
            ).all()
    
    assert(data['cell'] == torch.tensor([[0.2, 0.0, 0.0],
                                         [0.0, 0.2, 0.0],
                                         [0.0, 0.0, 0.2]])
          ).all()
    
    assert(data['node_attrs'] == torch.tensor([[0.0, 1.0],
                                               [1.0, 0.0], 
                                               [1.0, 0.0]])
           ).all()
    
    assert(data['node_labels'] == torch.tensor([[0.0], 
                                                [1.0], 
                                                [1.0]])
            ).all()
    
    assert(data['graph_labels'] == torch.tensor([[1.0]])).all()
    assert(data['weight'] == 1.0)

    # initialize configuration using two atoms (1 system, 1 env) as a subset
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
    
    # check edges and shift are computed correctly
    assert(data['edge_index'] == torch.tensor([[1, 2],
                                               [2, 1]])
            ).all()
    assert (data['shifts'] == torch.tensor([[0.0, 0.2, 0.0], 
                                            [0.0, -0.2, 0.0]])
            ).all()
    assert(data['unit_shifts'] == torch.tensor([[0.0, 1.0, 0.0], 
                                                [0.0, -1.0, 0.0]])
            ).all()

    # initialize configuration using three atoms (1 system, 2 env) as a subset and no buffer
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
    assert(data['edge_index'] == torch.tensor([[0, 0, 1, 1, 2, 2], 
                                                [2, 1, 0, 2, 1, 0]])
            ).all()


    # check if pbc and cutoffs works. now the third atoms is too far
    positions = np.array([[0.0, 0.0, 0.0], 
                          [0.07, 0.07, 0.0], 
                          [0.07, -0.08, 0.0]],
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
    # create dataset with same cutoff
    data = _create_dataset_from_configuration(config, z_table, 0.1)
    
    # check third atom is not included anymore
    assert (data['edge_index'] == torch.tensor([[0, 1], 
                                                [1, 0]])
        ).all()

    # create dataset with slightly large cutoff
    data = _create_dataset_from_configuration(config, z_table, 0.11)
    
    # check the edge with the third atom is created once again
    assert(data['edge_index'] == torch.tensor([[0, 0, 1, 1, 2, 2], 
                                                [2, 1, 0, 2, 1, 0]])
            ).all()
    
    # check with buffer layer
    # the third atoms should be included but with no edge to the system atom
    data = _create_dataset_from_configuration(config, z_table, 0.1, 0.01)
    assert(data['edge_index'] == torch.tensor([[0, 1, 1, 2], 
                                                [1, 0, 2, 1]])
            ).all()
    assert(data['shifts'] == torch.tensor([[0.0, 0.0, 0.0],        
                                            [0.0, 0.0, 0.0],        
                                            [0.0, 0.2, 0.0],        
                                            [0.0, -0.2, 0.0]])
          ).all()
    assert(data['unit_shifts'] == torch.tensor([[0.0, 0.0, 0.0],
                                                [0.0, 0.0, 0.0],
                                                [0.0, 1.0, 0.0],
                                                [0.0, -1.0, 0.0]])
           ).all()

    # create a list of configurations
    config = [atomic.Configuration(
        atomic_numbers=numbers,
        positions=positions,
        cell=cell,
        pbc=[True] * 3,
        node_labels=node_labels,
        graph_labels=np.array([[i]]),
    ) for i in range(0, 10)]

    # create dataset from list of configurations
    dataset = create_dataset_from_configurations(config, 
                                                 z_table, 
                                                 0.1, 
                                                 show_progress=False)

    # check if the labels of the entries are created correctly
    assert dataset.metadata['z_table'] == [1, 8]
    assert (dataset[0]['data_list']['graph_labels'] == torch.tensor([[0.0]])).all()
    assert (dataset[2]['data_list']['graph_labels'] == torch.tensor([[2.0]])).all()
    assert (dataset[4]['data_list']['graph_labels'] == torch.tensor([[4.0]])).all()

    # dataset_1 = dataset[np.array([0, -1])]
    assert dataset.metadata['z_table'] == [1, 8]
    assert (dataset[ 0]['data_list']['graph_labels'] == torch.tensor([[0.0]])).all()
    assert (dataset[-1]['data_list']['graph_labels'] == torch.tensor([[9.0]])).all()



def test_from_configurations() -> None:
    # fake atomic numbers, positions, cell, graph label, node labels
    numbers = [8, 1, 1]
    positions = np.array([[0.0, 0.0, 0.0], 
                          [0.07, 0.07, 0.0], 
                          [0.07, -0.07, 0.0]],
                         dtype=float
                        )
    cell = np.identity(3, dtype=float) * 0.2
    graph_labels = np.array([[1]])
    node_labels = np.array([[0], [1], [1]])

    # init AtomicNumber object
    z_table = atomic.AtomicNumberTable.from_zs(numbers)

    # initialize configuration using all atoms
    config = atomic.Configuration(
        atomic_numbers=numbers,
        positions=positions,
        cell=cell,
        pbc=[True] * 3,
        node_labels=node_labels,
        graph_labels=graph_labels,
    )

    # create dataset from a configuration, even if single is the multiple function
    dataset = create_dataset_from_configurations([config], 
                                              z_table, 
                                              0.1, 
                                              remove_isolated_nodes=True,
                                              show_progress=False
                                            )[0]
    
    # take data entry from the DictDataset
    data = dataset['data_list']

    # check edges and shifts are created correctly
    assert(data['edge_index'] == torch.tensor([[0, 0, 1, 1, 2, 2], 
                                               [2, 1, 0, 2, 1, 0]])
            ).all()
    assert(data['shifts'] == torch.tensor([[0.0, 0.0, 0.0],
                                           [0.0, 0.0, 0.0],
                                           [0.0, 0.0, 0.0],
                                           [0.0, 0.2, 0.0],
                                           [0.0, -0.2, 0.0],
                                           [0.0, 0.0, 0.0]])
            ).all()
    
    assert(data['unit_shifts'] == torch.tensor([[0.0, 0.0, 0.0],
                                                [0.0, 0.0, 0.0],
                                                [0.0, 0.0, 0.0],
                                                [0.0, 1.0, 0.0],
                                                [0.0, -1.0, 0.0],
                                                [0.0, 0.0, 0.0]])
            ).all()

    # check correct storage
    assert(data['positions'] == torch.tensor([[0.0, 0.0, 0.0],
                                              [0.07, 0.07, 0.0],
                                              [0.07, -0.07, 0.0]])
            ).all()
    
    assert(data['cell'] == torch.tensor([[0.2, 0.0, 0.0],
                                         [0.0, 0.2, 0.0],
                                         [0.0, 0.0, 0.2]])
            ).all()
    
    assert(data['node_attrs'] == torch.tensor([[0.0, 1.0], 
                                               [1.0, 0.0], 
                                               [1.0, 0.0]])
        ).all()
    assert(data['node_labels'] == torch.tensor([[0.0], 
                                                [1.0], 
                                                [1.0]])
            ).all()
    assert(data['graph_labels'] == torch.tensor([[1.0]])).all()
    assert(data['weight'] == 1.0)

    # initialize configuration using three atoms (1 system, 2 env) as a subset and no buffer
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
    dataset = create_dataset_from_configurations([config], 
                                              z_table, 
                                              0.1, 
                                              remove_isolated_nodes=True, 
                                              show_progress=False
                                            )[0]
    
    # take data entry from the DictDataset
    data = dataset['data_list']

    assert(data['positions'] == torch.tensor([[0.07, 0.07, 0.0], 
                                              [0.07, -0.07, 0.0]])
            ).all()
    assert(data['cell'] == torch.tensor([[0.2, 0.0, 0.0],
                                        [0.0, 0.2, 0.0],
                                        [0.0, 0.0, 0.2]])
            ).all()
    assert(data['node_attrs'] == torch.tensor([[1.0, 0.0], 
                                               [1.0, 0.0]])
            ).all()
    assert(data['edge_index'] == torch.tensor([[0, 1], 
                                               [1, 0]])
        ).all()
    assert(data['shifts'] == torch.tensor([[0.0, 0.2, 0.0], 
                                           [0.0, -0.2, 0.0]])
        ).all()
    assert(data['unit_shifts'] == torch.tensor([[0.0, 1.0, 0.0], 
                                                [0.0, -1.0, 0.0]])
        ).all()

if __name__ == '__main__':
    test_from_configuration()
    test_from_configurations()