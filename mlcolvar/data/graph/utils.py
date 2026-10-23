import copy
from collections import defaultdict
from typing import List, Union

import numpy as np
import torch
import torch_geometric
from torch_geometric.data import Data, HeteroData
from torch_geometric.transforms import BaseTransform

from mlcolvar.data import DictDataset, DictModule
from mlcolvar.data.graph import atomic
from mlcolvar.data.graph.neighborhood import get_neighborhood
from mlcolvar.utils.plot import pbar


__all__ = [
    "create_test_graph_input",
    "create_graph_tracing_example",
]

def _create_pyg_data_from_configuration(
    config: atomic.Configuration,
    atomic_numbers: atomic.AtomicNumberTable,
    cutoff: float,
    buffer: float = 0.0,
    long_range_cutoff: float = -1.0,
) -> torch_geometric.data.Data:
    """Build the torch_geometric graph data object from a configuration.

    Parameters
    ----------
    config: mlcolvar.data.graph.atomic.Configuration
        The configuration from which to generate the graph data
    atomic_numbers: mlcolvar.data.graph.atomic.AtomicNumberTable
        The atomic number table used to build the node attributes
    cutoff: float
        The graph cutoff radius
    buffer: float
        Buffer size used in finding active environment atoms if 
        restricting the neighborhood to a subsystem (i.e., system + environment), 
        `see also mlcolvar.data.grap.neighborhood.get_neighborhood`
    long_range_cutoff : float
            Cutoff radius for the long-range edges defined on subsystem atoms. 
            If negative, no long-range interactions are considered, by default -1.0
    """

    assert config.graph_labels is None or len(config.graph_labels.shape) == 2
    assert not ((config.subsystem is not None) ^ (long_range_cutoff > 0))

    # NOTE: here we do not take care about the nodes that are not taking part
    # the graph, like, we don't even change the node indices in `edge_index`.
    # Here we simply ignore them and rely on `RemoveIsolatedNodes`,
    # which is applied later during dataset preparation.
    edge_index, shifts, unit_shifts = get_neighborhood(
        positions=config.positions,
        cutoff=cutoff,
        cell=np.array(config.cell, copy=True),
        pbc=config.pbc,
        system_indices=config.system,
        environment_indices=config.environment,
        buffer=buffer
    )
    
    if config.subsystem is not None:
        config.subsystem = np.array(config.subsystem)
        edge_index_lr, shifts_lr, unit_shifts_lr = get_neighborhood(
            positions=config.positions[config.subsystem],
            cutoff=long_range_cutoff,
            cell=np.array(config.cell, copy=True),
            pbc=config.pbc,
        )
        edge_index_lr = np.vstack([config.subsystem[edge_index_lr[0]],
                                   config.subsystem[edge_index_lr[1]]])
        edge_index = np.hstack([edge_index, 
                                edge_index_lr])
        shifts = np.vstack([shifts,
                            shifts_lr])
        unit_shifts = np.vstack([unit_shifts,
                                 unit_shifts_lr])
    
    edge_index  = torch.tensor(edge_index, dtype=torch.long)
    shifts      = torch.tensor(shifts, dtype=torch.get_default_dtype())
    unit_shifts = torch.tensor(unit_shifts, dtype=torch.get_default_dtype())
    positions   = torch.tensor(config.positions, dtype=torch.get_default_dtype())
    cell        = torch.tensor(config.cell, dtype=torch.get_default_dtype())
    pbc         = torch.tensor(config.pbc, dtype=torch.bool).reshape(1, 3)
    
    node_labels  = torch.tensor( config.node_labels, dtype=torch.get_default_dtype() )    if config.node_labels is not None else None
    graph_labels = torch.tensor( config.graph_labels, dtype=torch.get_default_dtype() )   if config.graph_labels is not None else None
    weight       = torch.tensor( config.weight, dtype=torch.get_default_dtype() )         if config.weight is not None else 1

    # get indices from atomic numbers and convert to one_hot
    indices = atomic_numbers.zs_to_indices(config.atomic_numbers)
    one_hot = to_one_hot( torch.tensor( indices, dtype=torch.long ).unsqueeze(-1), n_classes=len(atomic_numbers) )

    # set n_system and system_mask
    if config.system is not None:
        n_system     = torch.tensor( [[len(config.system)]], dtype=torch.get_default_dtype() )
        system_masks = torch.zeros((one_hot.shape[0], 1), dtype=torch.bool)
        system_masks[config.system, 0] = 1
    else:
        n_system     = torch.tensor( [[one_hot.shape[0]]], dtype=torch.get_default_dtype() )
        system_masks = torch.ones((one_hot.shape[0], 1), dtype=torch.bool)

    # set subsystem_masks and edge_masks_lr
    if config.subsystem is not None:
        subsystem_masks = torch.zeros((one_hot.shape[0], 1), dtype=torch.bool)
        subsystem_masks[config.subsystem, 0] = 1
        edge_masks_lr = torch.zeros((edge_index.shape[1], 1), dtype=torch.bool)
        edge_masks_lr[-edge_index_lr.shape[1]:, 0] = 1
    else:
        subsystem_masks = torch.zeros((one_hot.shape[0], 1), dtype=torch.bool)
        edge_masks_lr = torch.zeros((edge_index.shape[1], 1), dtype=torch.bool)

    # set n_env
    n_env   = torch.tensor( [[one_hot.shape[0] - n_system.to(torch.int).item()]], dtype=torch.get_default_dtype() )


    pyg_data = torch_geometric.data.Data(edge_index=edge_index,
                                         shifts=shifts,
                                         unit_shifts=unit_shifts,
                                         positions=positions,
                                         cell=cell,
                                         pbc=pbc,
                                         node_attrs=one_hot,
                                         node_labels=node_labels,
                                         graph_labels=graph_labels,
                                         n_system=n_system,
                                         n_env=n_env,
                                         system_masks=system_masks,
                                         weight=weight,
                                         subsystem_masks=subsystem_masks,
                                         edge_masks_lr=edge_masks_lr,
                                        )
    
    return pyg_data


def _prepare_dataset_from_configurations(
    config: atomic.Configurations,
    atomic_numbers: atomic.AtomicNumberTable,
    cutoff: float,
    buffer: float = 0.0,
    long_range_cutoff: float = -1.0,
    atom_names: List = None,
    remove_isolated_nodes: bool = False,
    show_progress: bool = True,
):
    """Prepare graph data and metadata for dataset construction.

    This internal helper converts atomic configurations into
    ``torch_geometric`` graph objects and prepares the keyword arguments
    required to instantiate a graph-based ``DictDataset`` or compatible
    subclass.

    Parameters
    ----------
    config : mlcolvar.data.graph.atomic.Configurations
        Configurations from which to generate graph data.
    atomic_numbers : mlcolvar.data.graph.atomic.AtomicNumberTable
        Atomic number table used to build node attributes.
    cutoff : float
        Graph cutoff radius.
    buffer : float, optional
        Buffer used when selecting environment atoms around the system.
    long_range_cutoff : float, optional
        Cutoff radius for long-range edges defined on subsystem atoms.
        If negative, long-range interactions are not included.
    atom_names : list[str], optional
        Names of the system atoms.
    remove_isolated_nodes : bool, optional
        Whether to remove isolated nodes from the generated graphs.
    show_progress : bool, optional
        Whether to display the graph-construction progress bar.

    Returns
    -------
    dataset_kwargs : dict
        Keyword arguments required to instantiate a graph-based
        ``DictDataset`` or compatible subclass.
    """
    items = (
        pbar(config, frequency=0.0001, prefix="Making graphs")
        if show_progress
        else config
    )

    data_list = [
        _create_pyg_data_from_configuration(
            config=c,
            atomic_numbers=atomic_numbers,
            cutoff=cutoff,
            buffer=buffer,
            long_range_cutoff=long_range_cutoff,
        )
        for c in items
    ]

    try:
        truncated = any(c.environment.any() for c in config)
    except AttributeError:
        truncated = any(c.environment for c in config)

    if atom_names is None:
        n_system = data_list[0]["n_system"].to(torch.int64).item()
        atom_names = [f"X{i}" for i in range(n_system)]

    if remove_isolated_nodes:
        pre_remove_positions = torch.Tensor(
            np.array([d["positions"].numpy() for d in data_list])
        )
        pre_remove_system_masks = np.array(
            [d["system_masks"].numpy() for d in data_list],
            dtype=bool,
        )
        cell_list = [d.cell.clone() for d in data_list]

        transform = _RemoveIsolatedNodes()
        data_list = [transform(d) for d in data_list]

        unique_idx = []

        for i, data in enumerate(data_list):
            data.cell = cell_list[i]

            original_idx = torch.unique(
                torch.where(
                    torch.isin(
                        torch.round(
                            pre_remove_positions[i][
                                pre_remove_system_masks[i].squeeze(), :
                            ],
                            decimals=5,
                        ),
                        torch.round(
                            data["positions"][data["system_masks"].squeeze(), :],
                            decimals=5,
                        ),
                    )
                )[0]
            )

            data["system_names_idx"] = original_idx.to(torch.int64)

            check = np.isin(original_idx.numpy(), unique_idx, invert=True)
            if check.any():
                unique_idx.extend(original_idx[np.where(check)[0]].tolist())

        unique_idx = torch.tensor(sorted(unique_idx), dtype=torch.int64)

    else:
        unique_idx = torch.arange(
            data_list[0]["n_system"].item(),
            dtype=torch.int64,
        )

        for data in data_list:
            data["system_names_idx"] = unique_idx

    unique_names = [atom_names[i] for i in unique_idx.tolist()]

    return {
        "dictionary": {"data_list": data_list},
        "metadata": {
            "atomic_numbers": atomic_numbers.zs,
            "cutoff": cutoff,
            "buffer": buffer,
            "long_range_cutoff": long_range_cutoff,
            "system_idx": unique_idx,
            "system_atoms_names": unique_names,
            "is_truncated_graph": truncated,
        },
        "data_type": "graphs",
    }


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
                elif node_store.is_node_attr(key) and key not in [
                    'shifts',
                    'unit_shifts',
                    'cell',
                    'pbc',
                ]:
                    out[key] = value[n_id_dict[node_store._key]]

        return data


def create_test_graph_input(
    output_type: str,
    n_atoms: int = 3,
    n_samples: int = 60,
    n_states: int = 2,
    random_weights: bool = False,
    add_noise: bool = True,
    environment: bool = False,
    long_range: bool = False,
):
    """Create mock graph data for tests and examples.

    Parameters
    ----------
    output_type : str
        Output type: ``configuration``, ``configurations``, ``dataset``,
        ``datamodule``, ``batch``, or ``example``.
    n_atoms : int, optional
        Number of atoms, either 3 or 4.
    n_samples : int, optional
        Number of samples per state.
    n_states : int, optional
        Number of states.
    random_weights : bool, optional
        Whether to assign random sample weights.
    add_noise : bool, optional
        Whether to add small random noise to the reference positions.
    environment : bool, optional
        Whether to include environment atoms.
    long_range : bool, optional
        Whether to include long-range edges.
    """
    if n_atoms == 3:
        numbers = [8, 1, 1]
        system_atoms = [0, 1] if environment else None
        environment_atoms = [2] if environment else None
        buffer = 0.1 if environment else 0.0
        node_labels = np.array([[0], [1], [1]])
        _ref_positions = np.array(
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

    elif n_atoms == 4:
        numbers = [8, 1, 1, 8]
        system_atoms = [0,1,2] if environment else None
        environment_atoms = [3] if environment else None
        buffer = 0.1 if environment else 0.0
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

    subsystem_atoms = [0,1] if long_range else None

    np.random.seed(0)
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
    atomic_numbers = atomic.AtomicNumberTable.from_zs(numbers)

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
            weight=weights[i],
            system=system_atoms,
            environment=environment_atoms,
            subsystem=subsystem_atoms,
        ) for i in range(0, n_samples*n_states)
    ]

    if output_type == 'configuration':
        return config[0]
    if output_type == 'configurations':
        return config

    dataset = DictDataset.graph_from_configurations(
        config=config,
        atomic_numbers=atomic_numbers,
        cutoff=0.1,
        show_progress=False,
        remove_isolated_nodes=True,
        buffer=buffer,
        long_range_cutoff=0.3 if long_range else -1.0,
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

def create_graph_tracing_example(n_species: int,
                                 environment: bool = False,
                                 long_range: bool = False) -> dict:
    """
    Util to create a tracing example for graph based models.

    Parameters
    ----------
    n_species : int
        Number of chemical species to be considered in the model.
    environment : bool, optional
        Whether to include environment nodes in the graph, by default False.
    long_range : bool, optional
        Whether to include long-range edges in the graph, by default False.
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
    
    atomic_numbers = atomic.AtomicNumberTable.from_zs(numbers)
    
    system_atoms = [0,1] if environment else None
    environment_atoms = [2] if environment else None
    subsystem_atoms = [0,1] if long_range else None
    
    weights = np.ones((1, 1, 1))
    config = [
        atomic.Configuration(
            atomic_numbers=numbers,
            positions=positions[i],
            cell=cell,
            pbc=[True] * 3,
            node_labels=node_labels,
            graph_labels=graph_labels[i],
            weight=weights[i],
            system=system_atoms,
            environment=environment_atoms,
            subsystem=subsystem_atoms,
        ) for i in range(0, 1)
    ]

    # here we do not remove isolated nodes
    dataset = DictDataset.graph_from_configurations(
        config=config,
        atomic_numbers=atomic_numbers,
        cutoff=0.1,
        long_range_cutoff=0.3 if long_range else -1.0,
        show_progress=False,
        remove_isolated_nodes=False,
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