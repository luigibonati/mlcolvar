from typing import List, Any, Union
import numpy as np
from warnings import warn

from mlcolvar.data import DictDataset
from mlcolvar.io.graphs._utils import *
from mlcolvar.data.graph.atomic import Configuration, Configurations
from mlcolvar.data.graph.utils import create_dataset_from_configurations
from mlcolvar.io.graphs._utils import _to_torch_tensor

import ase
from ase.io import read, write
from ase import Atoms

__all__ = ["create_pdb_from_xyz", 
           "load_traj_with_ase",
           "dataset_from_ase_trajectories",
           "_configurations_from_ase_trajectory"]

def _prepare_configurations_from_ase_trajectories(
    trajectories: Union[
        List[ase.Atoms],
        List[List[ase.Atoms]],
    ],
    graph_labels: Union[list, List[list]] = None,
    node_labels: Union[list, List[list]] = None,
    system_selection=None,
    environment_selection=None,
    subsystem_selection=None,
    lengths_conversion: float = 1.0,
    atom_names: List = None,
):
    """Convert ASE trajectories into atomic configurations.

    Parameters
    ----------
    trajectories : List[ase.Atoms] or List[List[ase.Atoms]]
        ASE trajectory objects to convert.
    graph_labels : list, optional
        Frame-level graph labels for each trajectory.
    node_labels : list, optional
        Node-level labels for each trajectory.
    system_selection : optional
        Atom selection defining the system atoms. See Notes for supported formats.
    environment_selection : optional
        Atom selection defining the environment atoms. See Notes for supported formats.
    subsystem_selection : optional
        Atom selection defining the subsystem atoms. See Notes for supported formats.
    lengths_conversion : float, optional
        Length conversion factor, by default 1.0 because ASE uses Angstroms.
    atom_names : list, optional
        Optional names for system atoms. If not provided, names are
        inferred from the ASE Atoms objects.

    Returns
    -------
    configurations : Configurations
        Atomic configurations generated from all trajectory frames.
    atomic_numbers : AtomicNumberTable
        Atomic number table containing all species found in the trajectories.
    atom_names : list
        Names of the selected system atoms, either provided explicitly or
        inferred from the ASE Atoms objects.
    """

    if isinstance(trajectories, list):
        if isinstance(trajectories[0], ase.Atoms):
            trajectories = [trajectories]
    else:
        raise TypeError(
            "Trajectories must be a list of ase.Atoms or a list of lists!"
        )

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

    for i in range(len(trajectories)):
        configuration = _configurations_from_ase_trajectory(
            trajectory=trajectories[i],
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
        atom_names = _names_from_ase_atoms(
            ase_atoms_list=trajectories,
            system_selection=system_selection,
        )

    return configurations, atomic_numbers, atom_names


def dataset_from_ase_trajectories(
    trajectories: Union[
        List[ase.Atoms],
        List[List[ase.Atoms]],
    ],
    cutoff: float,
    graph_labels: Union[list, List[list]] = None,
    node_labels: Union[list, List[list]] = None,
    system_selection=None,
    environment_selection=None,
    subsystem_selection=None,
    lengths_conversion: float = 1.0,
    buffer: float = 0.0,
    long_range_cutoff: float = -1.0,
    atom_names: List = None,
    remove_isolated_nodes: bool = True,
    show_progress: bool = False,
) -> DictDataset:
    """Create a graph dataset from ASE trajectories.

    Parameters
    ----------
    trajectories : List[ase.Atoms] or List[List[ase.Atoms]]
        List of ASE trajectory frame sequences.
    cutoff : float
        Cutoff distance for graph edge construction in Angstroms.
    graph_labels : list, optional
        Frame-level graph labels for each trajectory.
    node_labels : list, optional
        Node-level labels for each trajectory.
    system_selection : optional
        ASE-style atom selection defining the system atoms.
    environment_selection : optional
        ASE-style atom selection defining the environment atoms.
    subsystem_selection : optional
        ASE-style atom selection defining the subsystem atoms.
    lengths_conversion : float, optional
        Length conversion factor, by default 1.0 because ASE uses Angstroms.
    buffer : float, optional
        Buffer used when selecting environment atoms.
    long_range_cutoff : float, optional
        Cutoff radius for long-range subsystem edges. If negative,
        long-range edges are not constructed.
    atom_names : list, optional
        Optional names for system atoms. If not provided, names are
        inferred from the ASE Atoms objects.
    remove_isolated_nodes : bool, optional
        Whether to remove isolated graph nodes.
    show_progress : bool, optional
        Whether to display graph-construction progress.

    Returns
    -------
    DictDataset
        Graph dataset created from the ASE trajectories.
        
    Notes
    -----
    Atom selection supports the following formats:

    - ``None``: select all atoms.
    - ``list``, ``tuple``, or ``np.ndarray`` of atom indices.
    - Boolean mask array-like.
    - ``callable(atoms)`` returning atom indices or a boolean mask.
    """

    _check_atom_selection(
        system_selection=system_selection,
        environment_selection=environment_selection,
        subsystem_selection=subsystem_selection,
        buffer=buffer,
        long_range_cutoff=long_range_cutoff,
    )

    (
        configurations,
        atomic_numbers,
        atom_names,
    ) = _prepare_configurations_from_ase_trajectories(
        trajectories=trajectories,
        graph_labels=graph_labels,
        node_labels=node_labels,
        system_selection=system_selection,
        environment_selection=environment_selection,
        subsystem_selection=subsystem_selection,
        lengths_conversion=lengths_conversion,
        atom_names=atom_names,
    )
    

    return create_dataset_from_configurations(
        config=configurations,
        atomic_numbers=atomic_numbers,
        cutoff=cutoff,
        buffer=buffer,
        long_range_cutoff=long_range_cutoff,
        atom_names=atom_names,
        remove_isolated_nodes=remove_isolated_nodes,
        show_progress=show_progress,
    )


def load_traj_with_ase(trajectory: str,
                       start: int = 0,
                       stop: int = None,
                       stride: int = 1) -> List[ase.Atoms]:
    """
    Load a trajectory using ASE.

    Parameters
    ----------
    trajectory : str
        Path to the trajectory file.
    start : int, optional
            Starting frame index, by default 0
    stop : int, optional
        Stopping frame index, by default None (load until the end)
    stride : int, optional
        Stride for frame selection, by default 1 (load all frames)    

    Returns
    -------
    List[ase.Atoms]
        Loaded ASE trajectory frames.
    """
    if stop is None:
        stop = ''
    frame_selection = f'{start}:{stop}:{stride}'
    
    # read trajectory with ASE
    traj = read(trajectory, index=frame_selection)
    
    
    return traj


def _selection_to_indices(selection, atoms):
    """Convert an ASE selection to a list of indices."""

    if selection is None:
        return np.arange(len(atoms)).tolist()
    
    if callable(selection):
        indices = np.asarray(selection(atoms))
    else:
        if isinstance(selection, str):
            raise TypeError("ASE selections do not support mdtraj-style selection strings. Use indices, boolean masks, or a callable instead.")
        indices = np.asarray(selection)

    if indices.dtype == bool:
        return np.nonzero(indices)[0].tolist()
    
    if indices.ndim == 0:
        return [int(indices)]
    
    return indices.tolist()


def _configurations_from_ase_trajectory(trajectory: List[ase.Atoms],
                                        graph_labels: list = None,
                                        node_labels: list = None,
                                        system_selection: Any = None,
                                        environment_selection: Any = None,
                                        subsystem_selection: Any = None,
                                        lengths_conversion: float = 1.0,
                                       ) -> Configurations:
    """Create configurations from one ASE trajectory frame sequence.

    Parameters
    ----------
    trajectory : ase.Atoms
        The ASE atoms object
    graph_labels : List, optional
        Frame-level graph labels for selected frames of this trajectory, by default None
    node_labels : List, optional
        Node-level graph labels for selected frames of this trajectory, by default None
    system_selection : Any, optional
        ASE style atom selection (see notes) of the system atoms, by default None. 
        If given, only selected atoms will be loaded from the trajectories into the configurations
        If not provided, all the atoms will be loaded.
    environment_selection : Any, optional
        ASE style atom selection (see notes) of the environment atoms, by default None. 
        If given, only the system atoms and the environment atoms will be included in the configuration.
    subsystem_selection : Any, optional
        ASE style atom selection (see notes) of the subsystem atoms for long-range interactions, by default None. 
    lengths_conversion : float, optional
        Conversion factor for length units, by default 1.
        The default corresponds to Angstroms which are already used by ASE.

    Returns
    -------
    Configurations
        List of the Configuration objects loaded from the trajectory

    Notes
    -------
    Atom selection can be done as in ASE. Supported formats are:
        - None: keep all atoms
        - list/tuple/np.ndarray of indices
        - boolean mask array-like
        - callable(atoms) -> indices
    """
    if isinstance(trajectory, Atoms):
        trajectory = [trajectory]

    # as we basically do the same for each selection, we use a dictionary initialized to the general case
    selected_atoms = {}
    selected_atoms['system'] = [i for i in range(len(trajectory[0]))]
    selected_atoms['environment'] = []
    selected_atoms['subsystem'] = None
    
    # here we only check if the selections are effective, compatibility has been checked above already
    for name, selection in {'system': system_selection, 
                            'environment': environment_selection, 
                            'subsystem': subsystem_selection}.items():
        if selection is not None:
            selected_atoms[name] = _selection_to_indices(selection, trajectory[0])
            if not len(selected_atoms[name]) > 0:
                raise ValueError(f"No atoms will be selected with selection {name}_selection: {selection}!")

    if subsystem_selection is not None:
        if not set(selected_atoms['subsystem']).issubset(set(selected_atoms['system'])):
            raise ValueError("Only atoms in `system_selection` can be selected by `subsystem_selection`!")


    # get the indeces of the required atoms
    selected_atoms['required'] = selected_atoms['system'] + selected_atoms['environment']
    
    # select the required atoms from the trajectory
    sliced_trajectory = []
    for frame in trajectory:
        sliced_trajectory.append(frame[selected_atoms['required']])
        
    # as we sliced the trajectory, we have to readjust the indeces to match the new order
    selected_atoms['system'] = np.arange(len(selected_atoms['system'])).tolist()
    selected_atoms['environment'] = (np.max(selected_atoms['system']) + 1 + np.arange(len(selected_atoms['environment'])) ).tolist()

    
    # get the list of the atomic numbers for the selected atoms
    atomic_numbers = sliced_trajectory[0].get_atomic_numbers().tolist()

    pbc = sliced_trajectory[0].get_pbc().tolist()

    if any(pbc):
        frame_cells = [frame.get_cell() for frame in sliced_trajectory]
    else:
        frame_cells = [None] * len(sliced_trajectory)


    # create configurations
    configurations = []
    for i in range(len(sliced_trajectory)):
        
        label_i = _to_torch_tensor(graph_labels[i]).reshape(-1, 1) if graph_labels is not None else None
        node_i = _to_torch_tensor(node_labels[i]).reshape(-1, 1) if node_labels is not None else None

        configuration = Configuration(atomic_numbers=atomic_numbers,
                                      positions=sliced_trajectory[i].get_positions() * lengths_conversion,
                                      cell=frame_cells[i] * lengths_conversion if frame_cells[i] is not None else None,
                                      pbc=pbc,
                                      graph_labels=label_i,
                                      node_labels=node_i,
                                      system=selected_atoms['system'],
                                      environment=selected_atoms['environment'],
                                      subsystem=selected_atoms['subsystem'],
        )
        configurations.append(configuration)


    return configurations


def _names_from_ase_atoms(
    ase_atoms_list: List[ase.Atoms],
    system_selection: Any,
) -> List[str]:
    """Create atomic names from a list of ASE Atoms objects."""
    try:
        indices = _selection_to_indices(system_selection, ase_atoms_list[0])
        names = ase_atoms_list[0][indices].get_chemical_symbols()
        
    except (AttributeError, TypeError):
        indices = _selection_to_indices(system_selection, ase_atoms_list[0][0])
        names = ase_atoms_list[0][0][indices].get_chemical_symbols()

    return names


def create_pdb_from_xyz(input_filename: str, output_filename: str) -> str:
    """
    Convert the first frame of an XYZ file into a PDB file using ASE.
    This pdb file can then serve as the topology for MDTraj.

    Parameters
    ----------
    input_filename : str
        Path to the input .xyz file.
    output_filename : str
        Path to the output .pdb file.

    Returns
    -------
    str
        The path to the generated PDB file.
    """

    atoms: Atoms = read(input_filename, index=0)

    if (atoms.cell == 0).all():
        warn("A topology file was generated from the xyz trajectory file but no cell information were provided!")
    if not atoms.pbc.any():
        warn("A topology file was generated from the xyz trajectory file but no PBC information were provided!")
    elif not atoms.pbc.all():
        warn( f"Partial PBC are not supported! The provided input has pbc {atoms.pbc}")

    write(output_filename, atoms, format='proteindatabank')

    return output_filename


def _get_cell_with_ase(trajectory):
    try:
        ase_atoms = read(trajectory, index=':')
        ase_cells = np.array([a.get_cell().array for a in ase_atoms], dtype=float)
        # the pdb for the topology are in nm, ase work in A so we need to scale it
        unitcell_vectors = ase_cells/10
    except Exception as e:
        warn(f"Could not load cell information with ASE for trajectory {trajectory}. Error: {e}")
        unitcell_vectors = None
    return unitcell_vectors