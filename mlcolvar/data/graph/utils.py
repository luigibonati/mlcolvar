import copy
from collections import defaultdict
from typing import Union

import torch
import torch_geometric
from torch_geometric.data import Data, HeteroData
from torch_geometric.transforms import BaseTransform

from mlcolvar.data import DictDataset, DictModule
from mlcolvar.data.graph import atomic
from mlcolvar.data.graph.neighborhood import get_neighborhood
from mlcolvar.utils.plot import pbar

from typing import List

__all__ = ["create_dataset_from_configurations", "create_test_graph_input"]

def _create_pyg_data_from_configuration(
    config: atomic.Configuration,
    z_table: atomic.AtomicNumberTable,
    cutoff: float,
    buffer: float = 0.0,
) -> torch_geometric.data.Data:
    """Build the torch_geometric graph data object from a configuration.

    Parameters
    ----------
    config: mlcolvar.data.graph.atomic.Configuration
        The configuration from which to generate the graph data
    z_table: mlcolvar.data.graph.atomic.AtomicNumberTable
        The atomic number table used to build the node attributes
    cutoff: float
        The graph cutoff radius
    buffer: float
        Buffer size used in finding active environment atoms if 
        restricting the neighborhood to a subsystem (i.e., system + environment), 
        `see also mlcolvar.data.grap.neighborhood.get_neighborhood`
    """

    assert config.graph_labels is None or len(config.graph_labels.shape) == 2

    # NOTE: here we do not take care about the nodes that are not taking part
    # the graph, like, we don't even change the node indices in `edge_index`.
    # Here we simply ignore them, and rely on the `RemoveIsolatedNodes` method
    # that will be called later (in `create_dataset_from_configurations`).
    edge_index, shifts, unit_shifts = get_neighborhood(positions=config.positions,
                                                       cutoff=cutoff,
                                                       cell=config.cell,
                                                       pbc=config.pbc,
                                                       system_indices=config.system,
                                                       environment_indices=config.environment,
                                                       buffer=buffer
                                                    )
    
    edge_index  = torch.tensor( edge_index, dtype=torch.long )
    shifts      = torch.tensor( shifts, dtype=torch.get_default_dtype() )
    unit_shifts = torch.tensor( unit_shifts, dtype=torch.get_default_dtype() )
    positions   = torch.tensor( config.positions, dtype=torch.get_default_dtype() )
    cell        = torch.tensor( config.cell, dtype=torch.get_default_dtype() )
    
    
    node_labels  = torch.tensor( config.node_labels, dtype=torch.get_default_dtype() )    if config.node_labels is not None else None
    graph_labels = torch.tensor( config.graph_labels, dtype=torch.get_default_dtype() )   if config.graph_labels is not None else None
    weight       = torch.tensor( config.weight, dtype=torch.get_default_dtype() )         if config.weight is not None else 1

    # get indices from atomic numbers and convert to one_hot
    indices = z_table.zs_to_indices(config.atomic_numbers)
    one_hot = to_one_hot( torch.tensor( indices, dtype=torch.long ).unsqueeze(-1), n_classes=len(z_table) )

    # set n_system and system_mask
    if config.system is not None:
        n_system     = torch.tensor( [[len(config.system)]], dtype=torch.get_default_dtype() )
        system_masks = torch.zeros((one_hot.shape[0], 1), dtype=torch.bool)
        system_masks[config.system, 0] = 1
    else:
        n_system     = torch.tensor( [[one_hot.shape[0]]], dtype=torch.get_default_dtype() )
        system_masks = None

    # set n_env
    n_env   = torch.tensor( [[one_hot.shape[0] - n_system.to(torch.int).item()]], dtype=torch.get_default_dtype() )


    pyg_data = torch_geometric.data.Data(edge_index=edge_index,
                                         shifts=shifts,
                                         unit_shifts=unit_shifts,
                                         positions=positions,
                                         cell=cell,
                                         node_attrs=one_hot,
                                         node_labels=node_labels,
                                         graph_labels=graph_labels,
                                         n_system=n_system,
                                         n_env=n_env,
                                         system_masks=system_masks,
                                         weight=weight,
                                        )
    
    return pyg_data


def create_dataset_from_configurations(config: atomic.Configurations,
                                       z_table: atomic.AtomicNumberTable,
                                       cutoff: float,
                                       buffer: float = 0.0,
                                       atom_names: List = None,
                                       remove_isolated_nodes: bool = False,
                                       show_progress: bool = True
                                      ) -> DictDataset:
    """Build DictDataset object containing torch_geometric graph data objects from configurations.

    Parameters
    ----------
    config: mlcolvar.graph.utils.atomic.Configurations
        The configurations from whihc to generate the dataset
    z_table: mlcolvar.graph.utils.atomic.AtomicNumberTable
        The atomic number table used to build the node attributes
    cutoff: float
        The graph cutoff radius
    buffer: float
        Buffer size used in finding active environment atoms if 
        restricting the neighborhood to a subsystem (i.e., system + environment), 
        `see also mlcolvar.data.grap.neighborhood.get_neighborhood`
    remove_isolated_nodes: bool
        If to remove isolated nodes from the dataset
    show_progress: bool
        If to show the progress bar
    """
    if show_progress:
        items = pbar(config, frequency=0.0001, prefix='Making graphs')
    else:
        items = config
    
    # create a list of torch_geometric data objects, one for each configuration
    data_list = [ _create_pyg_data_from_configuration(config=c, 
                                                      z_table=z_table, 
                                                      cutoff=cutoff, 
                                                      buffer=buffer, 
                                                     ) for c in items
                ]

    # get atom names if needed
    if atom_names is None:
        atom_names_system = [f"X{i}" for i in range(data_list[0]['n_system'].to(torch.int64).item())]
        atom_names_env    = [f"Y{i}" for i in range(data_list[0]['n_env'].to(torch.int64).item())]
        atom_names        = atom_names_system + atom_names_env


    if remove_isolated_nodes:
        # TODO: not the worst way to fake the `is_node_attr` method of `torch_geometric.data.storage.GlobalStorage`
        # If there are exact three atoms in the graph, the `RemoveIsolatedNodes` method will remove the cell vectors that
        # correspond to the isolated node. This is a consequence of pyg regarding the cell vectors as some kind of node features.
        # So here we first have to remove the isolated nodes and then set the cell back.
        
        # this aux var is only to check what isolated nodes have been removed
        _aux_pos = torch.Tensor((np.array([d['positions'].numpy() for d in data_list])))
        
        cell_list = [d.cell.clone() for d in data_list]
        transform = _RemoveIsolatedNodes()
        data_list = [transform(d) for d in data_list]
        
        # check what have been removed and restore cell
        unique_idx = [] # store the indeces of the atoms that have been used at least once
        for i in range(len(data_list)):
            data_list[i].cell = cell_list[i]

            # get and save the original index before removing isolated nodes for each entry
            original_idx = torch.unique( torch.where(torch.isin(torch.round(_aux_pos[i], decimals=5), 
                                                                torch.round(data_list[i]['positions'], decimals=5))
                                                    )[0]
                                        )
            
            data_list[i]['names_idx'] = original_idx.to(torch.int64)
            
            # update if needed the overall list
            check = np.isin(original_idx.numpy(), unique_idx, invert=True)
            if check.any():
                aux = np.where(check)[0]
                unique_idx.extend(original_idx[aux].tolist())
        
        unique_idx.sort()
        unique_idx = torch.Tensor(unique_idx).to(torch.int64)
    
    # if not remove_isolated_nodes we simply take all the atoms
    else:
        unique_idx = torch.arange(data_list[0]['n_system'].item()).to(torch.int64)
        for i in range(len(data_list)):
            data_list[i]['names_idx'] = unique_idx
    
    # we also save the names of the atoms that have been actually used
    unique_names = np.array(atom_names)[unique_idx]
    unique_names = unique_names.tolist()

    dataset = DictDataset(dictionary={'data_list': data_list},
                          metadata={'z_table': z_table.zs,
                                    'cutoff': cutoff,
                                    'used_idx': unique_idx,
                                    'used_names': unique_names},
                          data_type='graphs')

    return dataset


def to_one_hot(indices: torch.Tensor, n_classes: int) -> torch.Tensor:
    """Generates one-hot encoding with `n_classes` classes from `indices`

    Parameters
    ----------
    indices: torch.Tensor (shape: [N, 1])
        Node indices
    n_classes: int
        Number of classes

    Returns
    -------
    encoding: torch.tensor (shape: [N, n_classes])
        The one-hot encoding
    """
    shape = indices.shape[:-1] + (n_classes,)
    one_hot = torch.zeros(shape, device=indices.device).view(shape)

    # scatter_ is the in-place version of scatter
    one_hot.scatter_(dim=-1, index=indices, value=1)

    return one_hot.view(*shape)


class _RemoveIsolatedNodes(BaseTransform):
    r"""Removes isolated nodes from the graph
    This is taken from pytorch_geometric with a small modification to avoid the bug when n_nodes==n_edges
    """
    def forward(self,
                data: Union[Data, HeteroData],
               ) -> Union[Data, HeteroData]:
        """Remove isolated nodes from graphs in a pytorch_geometric Data object

        Parameters
        ----------
        data : Union[Data, HeteroData]
            Pytorch_geometric Data object containing the graph data

        Returns
        -------
        Union[Data, HeteroData]
            Pytorch_geometric Data object containing the modified graph data
        """
        # Gather all nodes that occur in at least one edge (across all types):
        n_ids_dict = defaultdict(list)
        for edge_store in data.edge_stores:
            if 'edge_index' not in edge_store:
                continue

            if edge_store._key is None:
                src = dst = None
            else:
                src, _, dst = edge_store._key

            n_ids_dict[src].append(edge_store.edge_index[0])
            n_ids_dict[dst].append(edge_store.edge_index[1])

        n_id_dict = {k: torch.cat(v).unique() for k, v in n_ids_dict.items()}

        n_map_dict = {}
        for node_store in data.node_stores:
            if node_store._key not in n_id_dict:
                n_id_dict[node_store._key] = torch.empty(0, dtype=torch.long)

            idx = n_id_dict[node_store._key]
            assert data.num_nodes is not None
            mapping = idx.new_zeros(data.num_nodes)
            mapping[idx] = torch.arange(idx.numel(), device=mapping.device)
            n_map_dict[node_store._key] = mapping

        for edge_store in data.edge_stores:
            if 'edge_index' not in edge_store:
                continue

            if edge_store._key is None:
                src = dst = None
            else:
                src, _, dst = edge_store._key

            row = n_map_dict[src][edge_store.edge_index[0]]
            col = n_map_dict[dst][edge_store.edge_index[1]]
            edge_store.edge_index = torch.stack([row, col], dim=0)

        old_data = copy.copy(data)
        for out, node_store in zip(data.node_stores, old_data.node_stores):
            for key, value in node_store.items():
                if key == 'num_nodes':
                    out.num_nodes = n_id_dict[node_store._key].numel()
                elif node_store.is_node_attr(key) and key not in ['shifts', 'unit_shifts']:
                    out[key] = value[n_id_dict[node_store._key]]

        return data


def create_test_graph_input(output_type: str,
                            n_atoms: int = 3,
                            n_samples: int = 60, 
                            n_states: int = 2,
                            random_weights = False,
                            add_noise = True):
    """
    Util function to generate several types of mock graph data objects for testing purposes.
    The graphs are created drawing positions from a predefined set of positions that cover most use cases.
    It can generate: one or some configuration objects, a dataset, a datamodule, a batch of example inputs or a single item.

    Parameters
    ----------
    output_type : str
        Type of graph data object to create. Can be: 'configuration', 'configurations', 'datamodule', 'dataset', 'batch', 'example'
    n_atoms : int, optional
        Number of atoms for creating the graph, either 3 or 4, by default 3
    n_samples : int, optional
        Number of samples per state to create, by default 60
    n_states : int, optional
        Number of states for which to create data, by default 2. Configurations are then labelled accordingly.
    random_weights : bool, optional
        If to assign random weights to the entries, otherwise unitary weights are given, by default False
    add_noise : bool, optional
        If to add a random noise for each entry to the predefined positions, by default True

    Returns
    -------
        Graph data object of the chosen type
    """
    if n_atoms == 3:
        numbers = [8, 1, 1]
        node_labels = np.array([[0], [1], [1]])
        _ref_positions = np.array(
            [
                [[0.0, 0.0, 0.0], [0.07, 0.07, 0.0], [0.07, -0.07, 0.0]],
                [[0.0, 0.0, 0.0], [-0.07, 0.07, 0.0], [0.07, 0.07, 0.0]],
                [[0.0, 0.0, 0.0], [0.07, -0.07, 0.0], [0.07, 0.07, 0.0]],
                [[0.0, 0.0, 0.0], [0.0, -0.07, 0.07], [0.0, 0.07, 0.07]],
                [[0.0, 0.0, 0.0], [0.11, 0.11, 0.11], [-0.07, 0.0, 0.07]],
                [[0.1, 0.0, 1.1], [0.17, 0.07, 1.1], [0.17, -0.07, 1.1]],
            ],
            dtype=np.float64
        )

    elif n_atoms == 4:
        numbers = [8, 1, 1, 8]
        node_labels = np.array([[0], [1], [1], [0]])
        _ref_positions = np.array(
            [
                [[0.0, 0.0, 0.0], [0.07, 0.07, 0.0] , [0.07, -0.07, 0.0], [0.05, -0.05, 0.0]],
                [[0.0, 0.0, 0.0], [-0.07, 0.07, 0.0], [0.07, 0.07, 0.0], [0.05, 0.05, 0.0]],
                [[0.0, 0.0, 0.0], [0.07, -0.07, 0.0], [0.07, 0.07, 0.0], [0.05, 0.05, 0.0]],
                [[0.0, 0.0, 0.0], [0.0, -0.07, 0.07], [0.0, 0.07, 0.07], [0.0, 0.05, 0.05]],
                [[0.0, 0.0, 0.0], [0.11, 0.11, 0.11] , [-0.07, 0.0, 0.07], [-0.05, 0.0, 0.05]],
                [[0.1, 0.0, 1.1], [0.17, 0.07, 1.1] , [0.17, -0.07, 1.1], [0.15, -0.05, 1.1]],
            ],
            dtype=np.float64
        )
    else:
        raise ValueError(f'Example input can be generated either with 3 or 4 atoms, found {n_atoms}')


    idx = np.random.randint(low=0, high=6, size=(n_samples*n_states))
    positions = _ref_positions[idx, :, :]

    # let's add some noise to the positions for fun
    if add_noise:
        noise = np.random.randn(*positions.shape)*1e-5
        positions = positions + noise

    cell = np.identity(3, dtype=float) * 0.2
    graph_labels = np.zeros((n_samples*n_states, 1, 1))
    for i in range(1, n_states):
            graph_labels[n_samples * i :] += 1
    z_table = atomic.AtomicNumberTable.from_zs(numbers)

    if random_weights:
        weights = np.random.random_sample((n_samples*n_states, 1, 1))
    else:
        weights = np.ones((n_samples*n_states, 1, 1))
    config = [
        atomic.Configuration(
            atomic_numbers=numbers,
            positions=positions[i],
            cell=cell,
            pbc=[True] * 3,
            node_labels=node_labels,
            graph_labels=graph_labels[i],
            weight=weights[i]
        ) for i in range(0, n_samples*n_states)
    ]

    if output_type == 'configuration':
        return config[0]
    if output_type == 'configurations':
        return config

    dataset = create_dataset_from_configurations(
        config, z_table, 0.1, show_progress=False, remove_isolated_nodes=True
    )

    if output_type == 'dataset':
        return dataset
    
    datamodule = DictModule(
        dataset,
        lengths=(0.8, 0.2),
        batch_size=0,
        shuffle=False,
    )

    if output_type == 'datamodule':
        return datamodule
    
    datamodule.setup()
    batch = next(iter(datamodule.train_dataloader()))
    if output_type == 'batch':
        return batch
    example = batch['data_list'].get_example(0)
    example['batch'] = torch.zeros(len(example['positions']), dtype=torch.int64)
    if output_type == 'example':
        return example
    
    return None

def create_graph_tracing_example(n_species : int):
    """
    Util to create a tracing example for graph based models.

    Parameters
    ----------
    n_species : int
        Number of chemical species to be considered in the model.

    Returns
    -------
    dict
        Tracing graph input example as dict.
    """
    numbers = [1, 1, 1]
    node_labels = np.array([[0], [0], [0]])
    _ref_positions = np.array(
        [
            [[0.0, 0.0, 0.0], [0.07, 0.07, 0.0], [0.07, -0.07, 0.0]],
            [[0.0, 0.0, 0.0], [-0.07, 0.07, 0.0], [0.07, 0.07, 0.0]],
            [[0.0, 0.0, 0.0], [0.07, -0.07, 0.0], [0.07, 0.07, 0.0]],
            [[0.0, 0.0, 0.0], [0.0, -0.07, 0.07], [0.0, 0.07, 0.07]],
            [[0.0, 0.0, 0.0], [0.11, 0.11, 0.11], [-0.07, 0.0, 0.07]],
            [[0.1, 0.0, 1.1], [0.17, 0.07, 1.1], [0.17, -0.07, 1.1]],
        ],
        dtype=np.float64
    )

    idx = np.random.randint(low=0, high=6, size=1)
    positions = _ref_positions[idx, :, :]
    cell = np.identity(3, dtype=float) * 0.2
    graph_labels = np.zeros((1, 1, 1))
    
    z_table = atomic.AtomicNumberTable.from_zs(numbers)

    weights = np.ones((1, 1, 1))
    config = [
        atomic.Configuration(
            atomic_numbers=numbers,
            positions=positions[i],
            cell=cell,
            pbc=[True] * 3,
            node_labels=node_labels,
            graph_labels=graph_labels[i],
            weight=weights[i]
        ) for i in range(0, 1)
    ]

    # here we do not remove isolated nodes
    dataset = create_dataset_from_configurations(
        config, z_table, 0.1, show_progress=False, remove_isolated_nodes=False
    )
    
    datamodule = DictModule(
        dataset,
        lengths=(0.8, 0.2),
        batch_size=0,
        shuffle=False,
    )
    
    datamodule.setup()
    batch = next(iter(datamodule.train_dataloader()))
    example = batch['data_list'].get_example(0)
    example['batch'] = torch.zeros(len(example['positions']), dtype=torch.int64)

    example = example.to_dict()
    example['node_attrs'] = torch.cat((example['node_attrs'], torch.zeros(3, n_species - 1)), 1)
    return example

# ===============================================================================
# ===============================================================================
# ==================================== TESTS ====================================
# ===============================================================================
# ===============================================================================

import numpy as np

def test_to_one_hot() -> None:
    i = torch.tensor([[0], [2], [1]], dtype=torch.int64)
    e = to_one_hot(i, 4)
    assert (
        e == torch.tensor(
            [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0]], dtype=torch.int64
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
    data = _create_pyg_data_from_configuration(config, z_table, 0.1)

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
    
    data = _create_pyg_data_from_configuration(config, z_table, 0.1)
    
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

    data = _create_pyg_data_from_configuration(config, z_table, 0.1)
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
    data = _create_pyg_data_from_configuration(config, z_table, 0.1)
    
    # check third atom is not included anymore
    assert (data['edge_index'] == torch.tensor([[0, 1], 
                                                [1, 0]])
        ).all()

    # create dataset with slightly large cutoff
    data = _create_pyg_data_from_configuration(config, z_table, 0.11)
    
    # check the edge with the third atom is created once again
    assert(data['edge_index'] == torch.tensor([[0, 0, 1, 1, 2, 2], 
                                                [2, 1, 0, 2, 1, 0]])
            ).all()
    
    # check with buffer layer
    # the third atoms should be included but with no edge to the system atom
    data = _create_pyg_data_from_configuration(config, z_table, 0.1, 0.01)
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
    test_to_one_hot()
    test_from_configuration()
    test_from_configurations()