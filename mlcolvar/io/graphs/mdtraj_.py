
from typing import List, Union
from warnings import warn

import mdtraj

from mlcolvar.data.graph.atomic import (
    AtomicNumberTable,
    Configuration,
    Configurations,
)
from mlcolvar.io.graphs._utils import (
    _format_labels,
    _update_atomic_numbers_from_configurations,
)
from mlcolvar.io.graphs.ase_ import _get_cell_with_ase


__all__ = [
    "load_traj_with_mdtraj",
    "_atomic_numbers_from_top",
    "_names_from_top",
]


def _prepare_configurations_from_mdtraj_trajectories(
    trajectories: List[mdtraj.Trajectory],
    graph_labels: list = None,
    node_labels: list = None,
    system_selection: str = None,
    environment_selection: str = None,
    subsystem_selection: str = None,
    lengths_conversion: float = 10.0,
    atom_names: List = None,
):
    """Convert MDTraj trajectories into atomic configurations.

    Parameters
    ----------
    trajectories : list[mdtraj.Trajectory]
        MDTraj trajectory objects to convert.
    graph_labels : list, optional
        Frame-level graph labels.
    node_labels : list, optional
        Node-level labels.
    system_selection : str, optional
        MDTraj-style atom selection defining the system atoms.
    environment_selection : str, optional
        MDTraj-style atom selection defining the environment atoms.
    subsystem_selection : str, optional
        MDTraj-style atom selection defining atoms used for long-range edges.
    lengths_conversion : float, optional
        Length conversion factor, by default 10 to convert nanometers
        to Angstroms.
    atom_names : list, optional
        Names of system atoms. If not provided, they are inferred from
        the MDTraj topologies.

    Returns
    -------
    configurations : Configurations
        Atomic configurations generated from all trajectory frames.
    atomic_numbers : AtomicNumberTable
        Atomic number table containing all species.
    atom_names : list
        Names of the selected system atoms.
    """
    graph_labels = _format_labels(
        trajectories=trajectories,
        labels=graph_labels,
    )
    node_labels = _format_labels(
        trajectories=trajectories,
        labels=node_labels,
    )

    configurations = []
    atomic_numbers = []

    for i, trajectory in enumerate(trajectories):
        configuration = _configurations_from_mdtraj_trajectory(
            trajectory=trajectory,
            graph_labels=graph_labels[i],
            node_labels=node_labels[i],
            system_selection=system_selection,
            environment_selection=environment_selection,
            subsystem_selection=subsystem_selection,
            lengths_conversion=lengths_conversion,
        )

        configurations.extend(configuration)
        atomic_numbers = _update_atomic_numbers_from_configurations(
            configurations=configuration,
            atomic_numbers=atomic_numbers,
        )

    if atom_names is None:
        atom_names = _names_from_top(
            top=[trajectory.topology for trajectory in trajectories],
            system_selection=system_selection,
        )

    return configurations, atomic_numbers, atom_names


def load_traj_with_mdtraj(
    trajectory: str,
    topology: str = None,
    start: int = 0,
    stop: int = None,
    stride: int = 1,
) -> mdtraj.Trajectory:
    """Load a trajectory using MDTraj.

    Parameters
    ----------
    trajectory : str
        Path to the trajectory file.
    topology : str
        Path to the topology file.
    start : int, optional
        Starting frame index.
    stop : int, optional
        Stopping frame index. If None, load until the end.
    stride : int, optional
        Stride for frame selection.

    Returns
    -------
    mdtraj.Trajectory
        Loaded and sliced MDTraj trajectory.
    """
    if not topology:
        raise ValueError(
            "MDTraj requires topology file(s) to load trajectories. "
            "For XYZ trajectories, create_pdb_from_xyz can be used "
            "to generate a topology file with ASE."
        )

    traj = mdtraj.load(trajectory, top=topology)
    traj.top = mdtraj.core.trajectory.load_topology(topology)

    if traj.unitcell_vectors is None:
        warn(
            f"Trajectory {trajectory} does not contain cell information "
            "that can be loaded by MDTraj. Trying ASE instead."
        )
        traj.unitcell_vectors = _get_cell_with_ase(trajectory)

        if traj.unitcell_vectors is None:
            raise ValueError(
                "Could not load cell information with either MDTraj or ASE. "
                "Check that the file contains cell information and that the "
                "format is supported by ASE."
            )

    if stop is None:
        stop = len(traj)

    return traj[list(range(start, stop, stride))]

def _configurations_from_mdtraj_trajectory(trajectory: mdtraj.Trajectory,
                                           graph_labels: list = None,
                                           node_labels: list = None,
                                           system_selection: str = None,
                                           environment_selection: str = None,
                                           subsystem_selection: str = None,
                                           lengths_conversion : float = 10.0) -> Configurations:
    """
    Create configurations from one trajectory.

    Parameters
    ----------
    trajectory: mdtraj.Trajectory
        The MDTraj Trajectory object.
    graph_labels: np.ndarray
        Frame-level graph labels for selected frames of this trajectory.
    node_labels : list, optional
        Node-level labels for selected frames of this trajectory.
    system_selection: str
        MDTraj style atom selection of the system atoms. If given, only
        selected atoms will be loaded from the trajectories. 
    environment_selection: str
        MDTraj style atom selection of the environment atoms. If given,
        only the system atoms and the environment atoms within the cutoff
        radius of the system atoms will be kept in the graph.
    subsystem_selection: str
        MDTraj style atom selection of the subsystem atoms for long-range interactions, by default None. 
    lengths_conversion: float,
        Conversion factor for length units, by default 10.
        MDTraj uses nanometers, the default sends to Angstroms.

    Returns
    -------
    Configurations
        List of the Configuration objects loaded from the trajectory
    """  
    
    # get the indeces of the required atoms (system + environment)
    required_atoms_selection = _get_required_atoms_selection(system_selection=system_selection,
                                                             environment_selection=environment_selection)
    

    # slice trajectory based on required selection
    subset = trajectory.top.select(required_atoms_selection)
    trajectory = trajectory.atom_slice(subset)

    # as we basically do the same for each selection (system, environment, subsystem) 
    # we use a dictionary initialized to the general case
    selected_atoms = {}
    selected_atoms['system'] = [i for i,e in enumerate(trajectory.top.atoms)]
    selected_atoms['environment'] = []
    selected_atoms['subsystem'] = None

    # here we only check if the selections are effective, compatibility has been checked above already
    for name, selection in {'system': system_selection, 
                            'environment': environment_selection, 
                            'subsystem': subsystem_selection}.items():
        if selection is not None:
            selected_atoms[name] = trajectory.top.select(selection)
            if not len(selected_atoms[name]) > 0:
                raise ValueError(f"No atoms will be selected with selection {name}_selection: {selection}!")
    

    if subsystem_selection is not None:
        if not set(selected_atoms['subsystem']).issubset(set(selected_atoms['system'])):
            raise ValueError("Only atoms in `system_selection` can be selected by `subsystem_selection`!")

    # get the list of the atomic numbers for the selected atoms
    atomic_numbers = [a.element.number for a in trajectory.top.atoms]
    
    if trajectory.unitcell_vectors is not None:
        pbc = [True] * 3
        cell = trajectory.unitcell_vectors
    else:
        pbc = [False] * 3
        cell = [None] * len(trajectory)

    
    # create configurations
    configurations = []
    for i in range(len(trajectory)):

        label_i = graph_labels[i].reshape(-1, 1) if graph_labels is not None else None
        node_i = node_labels[i].reshape(-1, 1) if node_labels is not None else None

        cell_i = (
            cell[i] * lengths_conversion
            if cell[i] is not None
            else None
        )

        configuration = Configuration(
            atomic_numbers=atomic_numbers,
            positions=trajectory.xyz[i] * lengths_conversion,
            cell=cell_i,
            pbc=pbc,
            graph_labels=label_i,
            node_labels=node_i,
            system=selected_atoms["system"],
            environment=selected_atoms["environment"],
            subsystem=selected_atoms["subsystem"],
        )
        
        configurations.append(configuration)

    return configurations


def _atomic_numbers_from_top(top: List[mdtraj.Topology]) -> AtomicNumberTable:
    """Create an atomic number table from the topologies."""

    atomic_numbers = []
    for t in top:
        atomic_numbers.extend([a.element.number for a in t.atoms])

    atomic_numbers = AtomicNumberTable.from_zs(atomic_numbers)

    return atomic_numbers


def _names_from_top(top: List[mdtraj.Topology],
                    system_selection: str) -> List[str]:
    """Retrieve atom names from the topologies."""
    
    if system_selection is None:
        system_selection = 'all'

    # apply selection
    top = [t.subset(t.select(system_selection)) for t in top]

    it = iter(top)
    atom_names = list(next(it).atoms)
    if not all([atom_names == list(n.atoms) for n in it]):
        raise ValueError("The atoms names or their order are different in the topology files. Check or deactivate save_names")
    return atom_names


def _get_required_atoms_selection(system_selection : str,
                                  environment_selection : str) -> str:
    """Define the selection string for the required atoms based on the system and environment selction"""

    if environment_selection is not None:
        required_atoms_selection = '({:s}) or ({:s})'.format(system_selection, environment_selection)
    elif system_selection is not None:
        required_atoms_selection = system_selection
    else:
        required_atoms_selection = 'all'
    return required_atoms_selection