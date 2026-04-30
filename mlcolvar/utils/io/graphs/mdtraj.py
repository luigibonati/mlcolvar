
import numpy as np
import os
from typing import  List
import mdtraj


from mlcolvar.utils.io.graphs._utils import *
from mlcolvar.data.graph.atomic import AtomicNumberTable, Configuration, Configurations
from mlcolvar.data.graph.utils import create_dataset_from_configurations

__all__ = ["load_traj_with_mdtraj", 
           "dataset_from_mdtraj_trajectories",
           "_z_table_from_top",
           "_names_from_top"]


def load_traj_with_mdtraj(trajectory: str, 
                          topology: str, 
                          selection: str):
    # load trajectory
        traj = mdtraj.load(trajectory, top=topology)
        traj.top = mdtraj.core.trajectory.load_topology(topology)
        
        # mdtraj does not load cell info from xyz, so we use ASE and add it
        _, ext = os.path.splitext(trajectory)
        if ext.lower() == ".xyz":
            try:
                from ase.io import read
            except ImportError as e:
                raise ImportError("ASE is required for creating the graph from a .xyz file.", e)
            ase_atoms = read(trajectory, index=':')
            ase_cells = np.array([a.get_cell().array for a in ase_atoms], dtype=float)
            # the pdb for the topology are in nm, ase work in A so we need to scale it
            traj.unitcell_vectors = ase_cells/10

        if selection is not None:
            subset = traj.top.select(selection)
            assert len(subset) > 0, (
                'No atoms will be selected with selection string '
                + '"{:s}"!'.format(selection)
            )
            traj = traj.atom_slice(subset)
        
        return traj

def dataset_from_mdtraj_trajectories(trajectories: List[mdtraj.Trajectory],
                                     graph_labels: List,
                                     node_labels: List,
                                     cutoff: float,
                                     z_table: AtomicNumberTable, 
                                     system_selection: str = None,
                                     environment_selection: str = None,
                                     subsystem_selection: str = None,
                                     load_args : dict = None,
                                     lengths_conversion : float = 10,
                                     buffer: float = 0.0,
                                     long_range_cutoff: float = -1.0,
                                     atom_names: List = None,
                                     remove_isolated_nodes: bool = False,
                                     show_progress: bool = True,
                                     ):
    # create configurations objects from trajectories
    configurations = []
    for i in range(len(trajectories)):
            configuration = _configurations_from_trajectory(
                trajectory=trajectories[i],
                graph_labels=graph_labels[i],
                node_labels=node_labels[i],
                system_selection=system_selection,
                environment_selection=environment_selection,
                subsystem_selection=subsystem_selection,
                start=load_args[i]['start'] if load_args is not None else 0,
                stop=load_args[i]['stop']  if load_args is not None else None,
                stride=load_args[i]['stride']  if load_args is not None else 1,
                lengths_conversion=lengths_conversion,
            )
            configurations.extend(configuration)

    # convert configurations into DictDataset
    dataset = create_dataset_from_configurations(
        config=configurations,
        z_table=z_table,
        cutoff=cutoff,
        buffer=buffer,
        long_range_cutoff=long_range_cutoff,
        atom_names=atom_names,
        remove_isolated_nodes=remove_isolated_nodes,
        show_progress=show_progress
    )
    return dataset

def _configurations_from_trajectory(
    trajectory: mdtraj.Trajectory,
    graph_labels = None,
    node_labels = None,
    system_selection: str = None,
    environment_selection: str = None,
    subsystem_selection: str = None,
    start: int = 0,
    stop: int = None,
    stride: int = 1,
    lengths_conversion : float = 10.0) -> Configurations:
    """
    Create configurations from one trajectory.

    Parameters
    ----------
    trajectory: mdtraj.Trajectory
        The MDTraj Trajectory object.
    graph_labels: np.ndarray
        Frame-level graph labels for selected frames of this trajectory.
    system_selection: str
        MDTraj style atom selections of the system atoms. If given, only
        selected atoms will be loaded from the trajectories. This option may
        increase the speed of building graphs.
    environment_selection: str
        MDTraj style atom selections of the environment atoms. If given,
        only the system atoms and [the environment atoms within the cutoff
        radius of the system atoms] will be kept in the graph.
    subsystem_selection: str
        MDTraj style atom selections of the subsystem atoms. If given, long
        edges will be put between subsystem atoms.
    lengths_conversion: float,
        Conversion factor for length units, by default 10.
        MDTraj uses nanometers, the default sends to Angstroms.
    """
    if system_selection is not None and environment_selection is not None:
        system_atoms = trajectory.top.select(system_selection)
        assert len(system_atoms) > 0, (
            'No atoms will be selected with `system_selection`: '
            + '"{:s}"!'.format(system_selection)
        )
        environment_atoms = trajectory.top.select(environment_selection)
        assert len(environment_atoms) > 0, (
            'No atoms will be selected with `environment_selection`: '
            + '"{:s}"!'.format(environment_selection)
        )
    else:
        system_atoms = None
        environment_atoms = None

    if subsystem_selection is not None:
        subsystem_atoms = trajectory.top.select(subsystem_selection)
        assert len(subsystem_atoms) > 0, (
            'No atoms will be selected with `subsystem_selection`: '
            + '"{:s}"!'.format(subsystem_selection)
        )
        # NOTE: in the above step we have done the atom_slice, so if no
        # environment atom has been defined, the subsystem atoms will
        # have to be selected from the sliced atoms, which are previously
        # defined by system_selection.
        # So here we only check if the subsystem_selection contains environment
        # atoms, under the case where both system_selection AND
        # environment_selection have been given.
        if system_selection is not None and environment_selection is not None:
            assert set(subsystem_atoms).issubset(set(system_atoms)), (
                "All atoms selected by `subsystem_selection` should also be "
                + "selected by `system_selection`!"
            )
    else:
        subsystem_atoms = None

    atomic_numbers = [a.element.number for a in trajectory.top.atoms]
    if trajectory.unitcell_vectors is not None:
        pbc = [True] * 3
        cell = trajectory.unitcell_vectors
    else:
        pbc = [False] * 3
        cell = [None] * len(trajectory)

    if stop is None:
        stop = len(trajectory)

    configurations = []
    frame_indices = list(range(start, stop, stride))

    for local_idx, i in enumerate(frame_indices):
        label_i = None
        if graph_labels is not None:
            label_i = _to_torch_tensor(graph_labels[local_idx]).reshape(-1, 1)

        node_i = None
        if node_labels is not None:
            node_i = _to_torch_tensor(node_labels[local_idx])
            if node_i.ndim == 1:
                node_i = node_i.reshape(-1, 1)

        configuration = Configuration(
            atomic_numbers=atomic_numbers,
            positions=trajectory.xyz[i] * lengths_conversion,
            cell=cell[i] * lengths_conversion,
            pbc=pbc,
            graph_labels=label_i,
            node_labels=node_i,
            system=system_atoms,
            environment=environment_atoms,
            subsystem=subsystem_atoms,
        )
        configurations.append(configuration)

    return configurations

def _z_table_from_top(
    top: List[mdtraj.Topology]
) -> AtomicNumberTable:
    """
    Create an atomic number table from the topologies.

    Parameters
    ----------
    top: List[mdtraj.Topology]
        The topology objects.
    """
    atomic_numbers = []
    for t in top:
        atomic_numbers.extend([a.element.number for a in t.atoms])
    # atomic_numbers = np.array(atomic_numbers, dtype=int)
    z_table = AtomicNumberTable.from_zs(atomic_numbers)
    return z_table

def _names_from_top(top: List[mdtraj.Topology] ):
    it = iter(top)
    atom_names = list(next(it).atoms)
    if not all([atom_names == list(n.atoms) for n in it]):
        raise ValueError(
            "The atoms names or their order are different in the topology files. Check or deactivate save_names"
        )
    
    return atom_names