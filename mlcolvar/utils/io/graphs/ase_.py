import os
from typing import List, Any
import numpy as np
from warnings import warn

from mlcolvar.data import DictDataset
from mlcolvar.utils.io.graphs._utils import *
from mlcolvar.data.graph.atomic import AtomicNumberTable, Configuration, Configurations
from mlcolvar.data.graph.utils import create_dataset_from_configurations
from mlcolvar.utils.io.graphs._utils import _to_torch_tensor

import ase
from ase.io import read, write
from ase import Atoms

__all__ = ["create_pdb_from_xyz", 
           "load_traj_with_ase",
           "dataset_from_ase_trajectories",
           "_configurations_from_ase_trajectory"]

def dataset_from_ase_trajectories(trajectories: List[ase.Atoms],
                                  graph_labels: List,
                                  node_labels: List,
                                  cutoff: float,
                                  system_selection=None,
                                  environment_selection=None,
                                  subsystem_selection=None,
                                  lengths_conversion: float = 1.0,
                                  buffer: float = 0.0,
                                  long_range_cutoff: float = -1.0,
                                  atom_names: List = None,
                                  remove_isolated_nodes: bool = False,
                                  show_progress: bool = True,
                                  ) -> DictDataset :
    """
    Create a graph dataset from ASE trajectories.

    Parameters
    ----------
    trajectories : List[ase.Atoms]
        List of ASE trajectory frame sequences loaded with ASE.
    graph_labels : List
        Frame-level graph labels for each trajectory.
    node_labels : List
        Node-level labels for each trajectory.
    cutoff : float
        Cutoff distance for graph edge construction (Angstroms).
    system_selection : optional
        ASE-style atom selection for the system atoms, by default None
    environment_selection : optional
        ASE-style atom selection for environment atoms, by default None
    subsystem_selection : optional
        ASE-style atom selection for subsystem atoms, by default None
    lengths_conversion : float, optional
        Length unit conversion factor, default 1.0 for ASE Angstroms
    buffer : float, optional
        Buffer size for truncated graph construction.
    long_range_cutoff : float, optional
        Long-range edge cutoff radius, by default -1.0 (no long-range edges). 
        If negative, long-range edges will not be constructed
    atom_names : List, optional
        Optional atom names used by the dataset constructor, by default None.
        If not provided, atomic numbers will be used to infer atom names from the ASE Atoms objects.
    remove_isolated_nodes : bool, optional
        Whether to remove isolated nodes from the final dataset, by default False
    show_progress : bool, optional
        Whether to show progress while building the dataset, by default True

    Notes
    -------
    Atom selection can be done as in ASE. Supported formats are:
        - None: keep all atoms
        - list/tuple/np.ndarray of indices
        - boolean mask array-like
        - callable(atoms) -> indices
    
    Returns
    -------
    DictDataset
         The graph dataset created from the MDtraj trajectories.
    """
    
    # Check compatibility of selection keywords combinations. NOTE: This doesn't check if the selection is correct.
    _check_atom_selection(system_selection=system_selection,
                          environment_selection=environment_selection,
                          subsystem_selection=subsystem_selection,
                          buffer=buffer,
                          long_range_cutoff=long_range_cutoff)
    
    # create configurations objects from trajectories
    configurations = []
    atomic_numbers = []
    for i in range(len(trajectories)):

        # TODO maybe this can be a single function with a backend argument
        # create configurations for this trajectory
        configuration = _configurations_from_ase_trajectory(trajectory=trajectories[i],
                                                            graph_labels=graph_labels[i],
                                                            node_labels=node_labels[i],
                                                            system_selection=system_selection,
                                                            environment_selection=environment_selection,
                                                            subsystem_selection=subsystem_selection,
                                                            lengths_conversion=lengths_conversion,
                                                           )
        # add configuration to configurations
        configurations.extend(configuration)
        
        # check if new atomic species have been discovered
        atomic_numbers = _update_atomic_numbers_from_configurations(configurations=configuration,
                                                                    atomic_numbers=atomic_numbers)
        
    # get atomic numbers and names from ASE Atoms objects if not provided
    if atom_names is None:
        atom_names = _names_from_ase_atoms(ase_atoms_list=trajectories,
                                           system_selection=system_selection)

    # create dataset from configurations list
    dataset = create_dataset_from_configurations(config=configurations,
                                                 atomic_numbers=atomic_numbers,
                                                 cutoff=cutoff,
                                                 buffer=buffer,
                                                 long_range_cutoff=long_range_cutoff,
                                                 atom_names=atom_names,
                                                 remove_isolated_nodes=remove_isolated_nodes,
                                                 show_progress=show_progress,
                                                )
    
    return dataset


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
        return None
    
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


# TODO maybe also this can framed into a shared function
def _configurations_from_ase_trajectory(trajectory: ase.Atoms,
                                        graph_labels: List = None,
                                        node_labels: List = None,
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
    selected_atoms['system'] = [i for i in range(trajectory[0].get_number_of_atoms())]
    selected_atoms['environment'] = []
    selected_atoms['subsystem'] = None
    
    # here we only check if the selections are effective, compatibility has been checked above already
    for name, selection in {'system': system_selection, 
                            'environment': environment_selection, 
                            'subsystem': subsystem_selection}.items():
        if selection is not None:
            # TODO maybe also this can framed into a shared function
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


def _names_from_ase_atoms(ase_atoms_list: List[ase.Atoms],
                          system_selection: Any) -> AtomicNumberTable:
    """Create atomic names from a list of ASE Atoms objects."""
    try:
        indices = _selection_to_indices(system_selection, ase_atoms_list[0])
        names = ase_atoms_list[0][indices].get_chemical_symbols()
    except AttributeError:
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