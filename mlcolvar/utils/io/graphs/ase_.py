import os
from typing import List
import numpy as np
from warnings import warn

from mlcolvar.data import DictDataset
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
                                  atomic_numbers: AtomicNumberTable,
                                  system_selection=None,
                                  environment_selection=None,
                                  subsystem_selection=None,
                                  load_args: dict = None,
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
    atomic_numbers : AtomicNumberTable
        Atomic number table used to build node features, by default None
    system_selection : optional
        ASE-style atom selection for the system atoms, by default None
    environment_selection : optional
        ASE-style atom selection for environment atoms, by default None
    subsystem_selection : optional
        ASE-style atom selection for subsystem atoms, by default None
    load_args : dict, optional
        Per-trajectory load arguments with keys start/stop/stride, by default None (load all frames)
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

    Returns
    -------
    DictDataset
         The graph dataset created from the MDtraj trajectories.
    """

    # get atomic numbers and names from ASE Atoms objects if not provided
    configurations = []
    # TODO move inside common?
    if atomic_numbers is None:
        atomic_numbers = _atomic_numbers_from_ase_atoms(trajectories)
    if atom_names is None:
        atom_names = _names_from_ase_atoms(trajectories)

    # convert trajectories into configurations    
    for i in range(len(trajectories)):
        # create configuration
        configuration = _configurations_from_ase_trajectory(trajectory=trajectories[i],
                                                            graph_labels=graph_labels[i],
                                                            node_labels=node_labels[i],
                                                            system_selection=system_selection,
                                                            environment_selection=environment_selection,
                                                            subsystem_selection=subsystem_selection,
                                                            start=load_args[i]['start'] if load_args is not None else 0,
                                                            stop=load_args[i]['stop'] if load_args is not None else None,
                                                            stride=load_args[i]['stride'] if load_args is not None else 1,
                                                            lengths_conversion=lengths_conversion,
                                                           )
        # add configuration to configurations
        configurations.extend(configuration)

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
                      topology: str = None,
                      selection=None) -> List[ase.Atoms]:
    """
    Load a trajectory using ASE.

    Parameters
    ----------
    trajectory : str
        Path to the trajectory file.
    topology : str, optional
        Unused for ASE-based loading, kept for API compatibility.
    selection : optional
        Atom selection for each frame. Supported formats are:
        - None: keep all atoms
        - list/tuple/np.ndarray of indices
        - boolean mask array-like
        - callable(atoms) -> indices

    Returns
    -------
    List[ase.Atoms]
        Loaded ASE trajectory frames.
    """

    # read trajectory with ASE
    traj = read(trajectory, index=':')
    if isinstance(traj, Atoms):
        traj = [traj]

    # apply selection if provided
    if selection is None:
        # simply return the original trajectory as a list of ASE Atoms objects
        return [traj]
    else:
        # for each frame in the traj we select the atoms according to the provided selection
        # and return a list of selected frames as ASE Atoms objects
        selected_frames = []
        for frame in traj:

            # convert callable into indices if needed
            indices = selection if not callable(selection) else selection(frame)
            
        #     if not isinstance(indices, (list, tuple)):
        #         indices = np.asarray(indices)
        #         if indices.dtype == bool:
        #             indices = indices.tolist()

        #     selected = frame[indices]

        #     selected_frames.append(selected)

        # return selected_frames

            if isinstance(indices, (list, tuple)):
                selected = frame[indices]
            else:
                indices = np.asarray(indices)
                
                if indices.dtype == bool:
                    selected = frame[indices]
                else:
                    selected = frame[indices.tolist()]

            selected_frames.append(selected)
        
        return selected_frames

def _selection_to_indices(selection, atoms):
    """Convert an ASE selection to a list of indices."""
    print(selection(atoms))

    if selection is None:
        return None
    
    if callable(selection):
        indices = selection(atoms)
    else:
        if isinstance(selection, str):
            raise TypeError(
                "ASE selections do not support mdtraj-style selection strings. "
                "Use indices, boolean masks, or a callable instead."
            )
        indices = np.asarray(selection)

    if indices.dtype == bool:
        return np.nonzero(indices)[0].tolist()
    
    if indices.ndim == 0:
        return [int(indices)]
    
    return indices.tolist()


def _configurations_from_ase_trajectory(trajectory,
                                        graph_labels=None,
                                        node_labels=None,
                                        system_selection=None,
                                        environment_selection=None,
                                        subsystem_selection=None,
                                        start: int = 0,
                                        stop: int = None,
                                        stride: int = 1,
                                        lengths_conversion: float = 1.0,
                                       ) -> Configurations:
    """Create configurations from one ASE trajectory frame sequence."""
    print(system_selection(trajectory[0]))

    # Check selections make sense
    if system_selection is not None:
        system_atoms = _selection_to_indices(system_selection, trajectory[0])
        print(system_atoms)
        assert len(system_atoms) > 0, (
            'No atoms will be selected with `system_selection`: '
            + '"{:s}"!'.format(str(system_selection))
        )

        if environment_selection is not None:
            environment_atoms = _selection_to_indices(environment_selection, trajectory[0])
            assert len(environment_atoms) > 0, (
                'No atoms will be selected with `environment_selection`: '
                + '"{:s}"!'.format(str(environment_selection))
            )
            assert set(system_atoms).isdisjoint(set(environment_atoms)), (
                "Atoms selected by `system_selection` and `environment_selection` should be disjoint!"
            )
        else:
            environment_atoms = []

    else:
        system_atoms = [_ for _ in range(len(trajectory[0]))]
        if environment_selection is not None:
            raise ValueError("`environment_selection` is provided without `system_selection`. Please provide a `system_selection` or remove the `environment_selection`.")

    print(system_atoms)

    if subsystem_selection is not None:
        subsystem_atoms = _selection_to_indices(subsystem_selection, trajectory[0])
        assert len(subsystem_atoms) > 0, (
            'No atoms will be selected with `subsystem_selection`: '
            + '"{:s}"!'.format(str(subsystem_selection))
        )
        if system_selection is not None and environment_selection is not None:
            assert set(subsystem_atoms).issubset(set(system_atoms)), (
                "All atoms selected by `subsystem_selection` should also be "
                + "selected by `system_selection`!"
            )
    else:
        subsystem_atoms = None
        
    atomic_numbers = trajectory[0].get_atomic_numbers().tolist()
    frame_cells = []
    pbc = trajectory[0].get_pbc().tolist()
    if any(pbc):
        frame_cells = [frame.get_cell() for frame in trajectory]
    else:
        frame_cells = [None] * len(trajectory)

    if stop is None:
        stop = len(trajectory)

    configurations = []
    frame_indices = list(range(start, stop, stride))

    for local_idx, i in enumerate(frame_indices):
        frame = trajectory[i]
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
            positions=frame.get_positions() * lengths_conversion,
            cell=frame_cells[i] * lengths_conversion if frame_cells[i] is not None else None,
            pbc=pbc,
            graph_labels=label_i,
            node_labels=node_i,
            system=system_atoms,
            environment=environment_atoms,
            subsystem=subsystem_atoms,
        )
        configurations.append(configuration)


    return configurations

def _atomic_numbers_from_ase_atoms(ase_atoms_list: List[ase.Atoms]) -> AtomicNumberTable:
    """Create an atomic number table from a list of ASE Atoms objects."""

    atomic_numbers = []
    for t in ase_atoms_list:
        atomic_numbers.extend([np.unique(a.get_atomic_numbers()).item() for a in t])
    
    atomic_numbers = AtomicNumberTable.from_zs(atomic_numbers)

    return atomic_numbers


def _names_from_ase_atoms(ase_atoms_list: List[ase.Atoms]) -> AtomicNumberTable:
    """Create atomic names from a list of ASE Atoms objects."""
    
    names = ase_atoms_list[0][0].get_chemical_symbols()

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



def test():
    from mlcolvar.tests import data_dir

    with data_dir() as data_folder:

        traj = load_traj_with_ase(trajectory=f"{data_folder}/Cu.xyz",
                                    topology=None,
                                    selection=None)
        

        dataset = dataset_from_ase_trajectories(trajectories=traj,
                                                graph_labels=[None for _ in traj],
                                                node_labels=[None for _ in traj],
                                                cutoff=3.5,  # Ang
                                                buffer=0.0,
                                                atomic_numbers=None,
                                                system_selection=lambda atoms: atoms.get_positions()[:, 2] < 0.1,
                                                environment_selection=None,
                                                show_progress=False,
                                                load_args=[{'start' : 0, 'stop' : 3, 'stride' : 1}],
                                            )

        print(dataset)                                            

if __name__ == '__main__':
    test()