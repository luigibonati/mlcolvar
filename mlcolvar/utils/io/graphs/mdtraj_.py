
import numpy as np
import os
from typing import  List
import mdtraj

from warnings import warn

from mlcolvar.data import DictDataset
from mlcolvar.utils.io.graphs._utils import *
from mlcolvar.utils.io.graphs.ase_ import _get_cell_with_ase
from mlcolvar.data.graph.atomic import AtomicNumberTable, Configuration, Configurations
from mlcolvar.data.graph.utils import create_dataset_from_configurations

__all__ = ["load_traj_with_mdtraj", 
           "dataset_from_mdtraj_trajectories",
           "_atomic_numbers_from_top",
           "_names_from_top"]


def dataset_from_mdtraj_trajectories(trajectories: List[mdtraj.Trajectory],
                                     graph_labels: List,
                                     node_labels: List,
                                     cutoff: float,
                                     atomic_numbers: AtomicNumberTable, 
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
                                     ) -> DictDataset:
    """    
    Create a graph dataset from MDtraj trajectories.


    Parameters
    ----------
    trajectories : List[mdtraj.Trajectory]
        List of MDtraj trajectory frame sequences loaded with MDtraj.
    graph_labels : List
        Frame-level graph labels for each trajectory.
    node_labels : List
        Node-level labels for each trajectory.
    cutoff : float
        Cutoff distance for graph edge construction (Angstroms).
    atomic_numbers : AtomicNumberTable
        Atomic number table used to build node features.
    system_selection : str, optional
        MDtraj style atom selection for the system atoms, by default None
    environment_selection : str, optional
        MDtraj style atom selection for the environment atoms, by default None
    subsystem_selection : str, optional
        MDtraj style atom selection for the subsystem atoms, by default None
    load_args : dict, optional
        Per-trajectory load arguments with keys start/stop/stride, by default None
    lengths_conversion : float, optional
        Length unit conversion factor, by default 10 for MDtraj nanometers to Angstroms
    buffer : float, optional
        Buffer size for truncated graph construction.
    long_range_cutoff : float, optional
        Long-range edge cutoff radius, by default -1.0 (no long-range edges). 
        If negative, long-range edges will not be constructed
    atom_names : List, optional
        Optional atom names used by the dataset constructor, by default None
        If not provided, atomic names will be infered from the MDtraj Topology objects
    remove_isolated_nodes : bool, optional
        Whether to remove isolated nodes from the final dataset, by default False
    show_progress : bool, optional
        Whether to show progress while building the dataset, by default True

    Returns
    -------
    DictDataset
         The graph dataset created from the MDtraj trajectories.
    """

    # create configurations objects from trajectories
    configurations = []
    for i in range(len(trajectories)):
        configuration = _configurations_from_mdtraj_trajectory(trajectory=trajectories[i],
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

    # create dataset from configurations list
    dataset = create_dataset_from_configurations(config=configurations,
                                                 atomic_numbers=atomic_numbers,
                                                 cutoff=cutoff,
                                                 buffer=buffer,
                                                 long_range_cutoff=long_range_cutoff,
                                                 atom_names=atom_names,
                                                 remove_isolated_nodes=remove_isolated_nodes,
                                                 show_progress=show_progress
                                               )
    
    return dataset


def load_traj_with_mdtraj(trajectory: str, 
                          topology: str, 
                          ) -> List[mdtraj.Trajectory]:
        """
        Load a trajectory using MDtraj.

        Parameters
        ----------
        trajectory : str
            Path to the trajectory file.
        topology : str, optional
            Path to the topology file.
        Returns
        -------
        List[mdtraj.Trajectory]
            Loaded MDtraj trajectory frames.
        """

        # load trajectory with mdtraj
        traj = mdtraj.load(trajectory, top=topology)
        traj.top = mdtraj.core.trajectory.load_topology(topology)
        
        # mdtraj does not load cell info from certain file types
        # so we try use ASE and add it
        if traj.unitcell_vectors is None:
            warn (f"Trajectory {trajectory} does not contain cell information that can be loaded by MDtraj. Trying to load cell information with ASE...")
            traj.unitcell_vectors = _get_cell_with_ase(trajectory)
            if traj.unitcell_vectors is None:
                raise ValueError(
                    f"Could not load cell information with neither MDtraj nor ASE from this file format. Check if the file contains cell information and if the file format is supported by ASE."
                )
        
        # # apply selection if provided
        # if selection is not None:
        #     subset = traj.top.select(selection)
        #     assert len(subset) > 0, (
        #         'No atoms will be selected with selection string '
        #         + '"{:s}"!'.format(selection)
        #     )
        #     traj = traj.atom_slice(subset)
        
        return traj

def _configurations_from_mdtraj_trajectory(
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

    # check if using truncated graph
    if environment_selection is not None:
        assert system_selection is not None, (
            'the `environment_selection` argument requires the'
            + '`system_selection` argument to be defined!'
        )
        selection = '({:s}) or ({:s})'.format(
            system_selection, environment_selection
        )
    elif system_selection is not None:
        selection = system_selection
    else:
        selection = 'all'
 
    subset = trajectory.top.select(selection)
    trajectory = trajectory.atom_slice(subset)

    system_atoms = [i for i,e in enumerate(trajectory.top.atoms)]

    environment_atoms = []  


    if system_selection is not None:
        system_atoms = trajectory.top.select(system_selection)
        assert len(system_atoms) > 0, (
            'No atoms will be selected with `system_selection`: '
            + '"{:s}"!'.format(system_selection)
        )

        if environment_selection is not None:
            environment_atoms = trajectory.top.select(environment_selection)
            assert len(environment_atoms) > 0, (
                'No atoms will be selected with `environment_selection`: '
                + '"{:s}"!'.format(environment_selection)
            )
    elif environment_selection is not None:
            raise ValueError("`environment_selection` is provided without `system_selection`. Please provide a `system_selection` or remove the `environment_selection`.")

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

    atomic_numbers = [a.element.number for a in trajectory.top.atoms] #trajectory.top.atoms]
    
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

def _atomic_numbers_from_top(
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
    atomic_numbers = AtomicNumberTable.from_zs(atomic_numbers)
    return atomic_numbers

def _names_from_top(top: List[mdtraj.Topology] ):
    it = iter(top)
    atom_names = list(next(it).atoms)
    if not all([atom_names == list(n.atoms) for n in it]):
        raise ValueError(
            "The atoms names or their order are different in the topology files. Check or deactivate save_names"
        )
    
    return atom_names