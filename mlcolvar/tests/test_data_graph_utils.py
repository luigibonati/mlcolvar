import numpy as np
import torch

from mlcolvar.data import DictDataset
from mlcolvar.data.graph import atomic
from mlcolvar.data.graph.utils import (
    _create_pyg_data_from_configuration,
    to_one_hot,
)


def test_to_one_hot() -> None:
    i = torch.tensor([[0], [2], [1]], dtype=torch.int64)
    e = to_one_hot(i, 4)

    assert (
        e
        == torch.tensor(
            [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0]],
            dtype=torch.int64,
        )
    ).all()

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
    atomic_numbers = atomic.AtomicNumberTable.from_zs(numbers)

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
    data = _create_pyg_data_from_configuration(config, atomic_numbers, 0.1)

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
    
    data = _create_pyg_data_from_configuration(config, atomic_numbers, 0.1)
    
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

    data = _create_pyg_data_from_configuration(config, atomic_numbers, 0.1)
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
    data = _create_pyg_data_from_configuration(config, atomic_numbers, 0.1)
    
    # check third atom is not included anymore
    assert (data['edge_index'] == torch.tensor([[0, 1], 
                                                [1, 0]])
        ).all()

    # create dataset with slightly large cutoff
    data = _create_pyg_data_from_configuration(config, atomic_numbers, 0.11)
    
    # check the edge with the third atom is created once again
    assert(data['edge_index'] == torch.tensor([[0, 0, 1, 1, 2, 2], 
                                                [2, 1, 0, 2, 1, 0]])
            ).all()
    
    # check with buffer layer
    # the third atoms should be included but with no edge to the system atom
    data = _create_pyg_data_from_configuration(config, atomic_numbers, 0.1, 0.01)
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
    dataset = DictDataset.graph_from_configurations(
        config=config,
        atomic_numbers=atomic_numbers,
        cutoff=0.1,
        show_progress=False,
    )

    # check if the labels of the entries are created correctly
    assert dataset.metadata['atomic_numbers'] == [1, 8]
    assert (dataset[0]['data_list']['graph_labels'] == torch.tensor([[0.0]])).all()
    assert (dataset[2]['data_list']['graph_labels'] == torch.tensor([[2.0]])).all()
    assert (dataset[4]['data_list']['graph_labels'] == torch.tensor([[4.0]])).all()

    # dataset_1 = dataset[np.array([0, -1])]
    assert dataset.metadata['atomic_numbers'] == [1, 8]
    assert (dataset[ 0]['data_list']['graph_labels'] == torch.tensor([[0.0]])).all()
    assert (dataset[-1]['data_list']['graph_labels'] == torch.tensor([[9.0]])).all()


def test_from_configuration_long_cutoff() -> None:
    # fake atomic numbers, positions, cell, graph label, node labels
    numbers = [8, 1, 1]
    positions = np.array([[0.0, 0.0, 0.0], 
                          [0.07, 0.07, 0.0], 
                          [0.07, -0.08, 0.0]],
                        dtype=float
                        )
    cell = np.identity(3, dtype=float) * 0.2
    graph_labels = np.array([[1]])
    node_labels = np.array([[0], [1], [1]])

    # init AtomicNumber object
    atomic_numbers = atomic.AtomicNumberTable.from_zs(numbers)

    # initialize configuration using all atoms
    config = atomic.Configuration(
        atomic_numbers=numbers,
        positions=positions,
        cell=cell,
        pbc=[True] * 3,
        node_labels=node_labels,
        graph_labels=graph_labels,
        system=[1, 2],
        environment=[0],
        subsystem=[1, 2],
    )

    # create dataset from a configuration
    data = _create_pyg_data_from_configuration(
        config, atomic_numbers, 0.1, long_range_cutoff=0.11
    )

    # check edges and shifts are created correctly
    assert (
        data['edge_index'] == torch.tensor(
            [[0, 1, 1, 2, 1, 2], [1, 0, 2, 1, 2, 1]]
        )
    ).all()
    assert (
        data['shifts'] == torch.tensor([
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.2, 0.0],
            [0.0, -0.2, 0.0],
            [0.0, 0.2, 0.0],
            [0.0, -0.2, 0.0],
        ])
    ).all()
    assert (
        data['unit_shifts'] == torch.tensor([
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
        ])
    ).all()
    assert (
        data['positions'] == torch.tensor([
            [0.0, 0.0, 0.0],
            [0.07, 0.07, 0.0],
            [0.07, -0.08, 0.0],
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
    assert (data['edge_masks_lr'] == torch.tensor([[0]] * 4 + [[1]] * 2)).all()
    assert (data['node_labels'] == torch.tensor([[0.0], [1.0], [1.0]])).all()
    assert (data['graph_labels'] == torch.tensor([[1.0]])).all()
    assert (data['edge_masks_lr'] == torch.tensor([[0]] * 4 + [[1]] * 2)).all()
    assert (data['system_masks'] == torch.tensor([[0], [1], [1]])).all()
    assert (data['subsystem_masks'] == torch.tensor([[0], [1], [1]])).all()
    assert data['weight'] == 1.0

    # initialize configuration using all atoms
    config = atomic.Configuration(
        atomic_numbers=numbers,
        positions=positions,
        cell=cell,
        pbc=[True] * 3,
        node_labels=node_labels,
        graph_labels=graph_labels,
        system=[0, 2],
        environment=[1],
        subsystem=[0, 2],
    )

    # create dataset from a configuration
    data = _create_pyg_data_from_configuration(
        config, atomic_numbers, 0.1, long_range_cutoff=0.11
    )

    # check edges and shifts are created correctly
    assert (
        data['edge_index'] == torch.tensor(
            [[0, 1, 1, 2, 0, 2], [1, 0, 2, 1, 2, 0]]
        )
    ).all()
    assert (data['edge_masks_lr'] == torch.tensor([[0]] * 4 + [[1]] * 2)).all()
    assert (data['system_masks'] == torch.tensor([[1], [0], [1]])).all()
    assert (data['subsystem_masks'] == torch.tensor([[1], [0], [1]])).all()

    # fake atomic numbers, positions, cell, graph label, node labels
    numbers = [8, 1, 1, 8, 1, 1]
    positions = np.array(
        [
            [0.0, 0.0, 0.0], [0.07, 0.07, 0.0], [0.07, -0.07, 0.0],
            [0.0, 0.8, 0.0], [0.07, 0.88, 0.0], [0.07, 0.73, 0.0],
        ],
        dtype=float
    )
    cell = np.identity(3, dtype=float)
    graph_labels = np.array([[1]])
    node_labels = np.array([[0], [1], [1], [0], [1], [1]])
    atomic_numbers = atomic.AtomicNumberTable.from_zs(numbers)

    # initialize configuration using all atoms
    config = atomic.Configuration(
        atomic_numbers=numbers,
        positions=positions,
        cell=cell,
        pbc=[True] * 3,
        node_labels=node_labels,
        graph_labels=graph_labels,
        system=[0, 3],
        environment=[1, 2, 4, 5],
        subsystem=[0, 3],
    )

    # create dataset from a configuration
    data = _create_pyg_data_from_configuration(
        config, atomic_numbers, 0.1, long_range_cutoff=0.4
    )

    # check edges and shifts are created correctly
    assert (
        data['edge_index'] == torch.tensor([
            [[0, 0, 1, 2, 3, 5, 0, 3], [2, 1, 0, 0, 5, 3, 3, 0]]
        ])
    ).all()
    assert (
        data['shifts'] == torch.tensor([
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 1.0, 0.0],
        ])
    ).all()
    assert (
        data['unit_shifts'] == torch.tensor([
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 1.0, 0.0],
        ])
    ).all()
    assert (data['edge_masks_lr'] == torch.tensor(
        [[0]] * 6 + [[1]] * 2
    )).all()
    assert (data['subsystem_masks'] == torch.tensor(
        [[1], [0], [0], [1], [0], [0]]
    )).all()

    # create dataset from a configuration
    data = _create_pyg_data_from_configuration(
        config, atomic_numbers, 0.1, buffer=0.011, long_range_cutoff=0.4
    )

    # check edges and shifts are created correctly
    assert (
        data['edge_index'] == torch.tensor([
            [0, 0, 1, 2, 2, 3, 4, 5, 0, 3],
            [2, 1, 0, 4, 0, 5, 2, 3, 3, 0]
        ])
    ).all()
    assert (
        data['shifts'] == torch.tensor([
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 1.0, 0.0],
        ])
    ).all()
    assert (
        data['unit_shifts'] == torch.tensor([
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 1.0, 0.0],
        ])
    ).all()
    assert (data['edge_masks_lr'] == torch.tensor(
        [[0]] * 8 + [[1]] * 2
    )).all()


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
    atomic_numbers = atomic.AtomicNumberTable.from_zs(numbers)

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
    dataset = DictDataset.graph_from_configurations(
        config=[config],
        atomic_numbers=atomic_numbers,
        cutoff=0.1,
        remove_isolated_nodes=True,
        show_progress=False,
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
    dataset = DictDataset.graph_from_configurations(
        config=[config],
        atomic_numbers=atomic_numbers,
        cutoff=0.1,
        remove_isolated_nodes=True,
        show_progress=False,
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


def test_from_configurations_long_cutoff() -> None:
    # fake atomic numbers, positions, cell, graph label, node labels
    numbers = [8, 1, 1, 8, 1, 1]
    positions = np.array(
        [
            [0.0, 0.0, 0.0], [0.07, 0.07, 0.0], [0.07, -0.07, 0.0],
            [0.0, 0.0, 0.9], [0.07, 0.08, 0.9], [0.07, -0.07, 0.9],
        ],
        dtype=float
    )
    cell = np.identity(3, dtype=float) * 1.1
    graph_labels = np.array([[1]])
    node_labels = np.array([[0], [1], [1], [0], [1], [1]])

    # init AtomicNumber object
    atomic_numbers = atomic.AtomicNumberTable.from_zs(numbers)

    # initialize configuration using all atoms
    config = atomic.Configuration(
        atomic_numbers=numbers,
        positions=positions,
        cell=cell,
        pbc=[True] * 3,
        node_labels=node_labels,
        graph_labels=graph_labels,
        system=[0, 3],
        environment=[1, 2, 4, 5],
        subsystem=[0, 3],
    )

    # create dataset from a configuration, even if single is the multiple function
    dataset = DictDataset.graph_from_configurations(
        config=[config],
        atomic_numbers=atomic_numbers,
        cutoff=0.11,
        long_range_cutoff=0.4,
        remove_isolated_nodes=True,
        show_progress=False,
    )[0]

    # take data entry from the DictDataset
    data = dataset['data_list']

    # check edges and shifts are created correctly
    assert (
        data['edge_index'] == torch.tensor([
            [[0, 0, 1, 2, 3, 3, 4, 5, 0, 3], [2, 1, 0, 0, 5, 4, 3, 3, 3, 0]]
        ])
    ).all()
    assert (
        data['shifts'] == torch.tensor([
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, -1.1],
            [0.0, 0.0, 1.1],
        ])
    ).all()
    assert (
        data['unit_shifts'] == torch.tensor([
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
            [0.0, 0.0, 1.0],
        ])
    ).all()
    assert (data['edge_masks_lr'] == torch.tensor(
        [[0]] * 8 + [[1]] * 2
    )).all()

    # create dataset from a configuration, even if single is the multiple function
    dataset = DictDataset.graph_from_configurations(
        config=[config],
        atomic_numbers=atomic_numbers,
        cutoff=0.1,
        long_range_cutoff=0.4,
        remove_isolated_nodes=True,
        show_progress=False,
    )[0]

    # take data entry from the DictDataset
    data = dataset['data_list']

    # check edges and shifts are created correctly
    assert (
        data['edge_index'] == torch.tensor([
            [[0, 0, 1, 2, 3, 4, 0, 3], [2, 1, 0, 0, 4, 3, 3, 0]]
        ])
    ).all()
    assert (
        data['shifts'] == torch.tensor([
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, -1.1],
            [0.0, 0.0, 1.1],
        ])
    ).all()
    assert (
        data['unit_shifts'] == torch.tensor([
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
            [0.0, 0.0, 1.0],
        ])
    ).all()
    assert (data['edge_masks_lr'] == torch.tensor(
        [[0]] * 6 + [[1]] * 2
    )).all()
    assert (
        data['positions'] == torch.tensor([
            [0.0, 0.0, 0.0],
            [0.07, 0.07, 0.0],
            [0.07, -0.07, 0.0],
            [0.0, 0.0, 0.9],
            [0.07, -0.07, 0.9]
        ])
    ).all()