from typing import Any, List, Union
from warnings import warn

import ase
import numpy as np
from ase import Atoms
from ase.io import read, write

from mlcolvar.data.graph.atomic import Configuration, Configurations
from mlcolvar.io.graphs._utils import (
    _format_labels,
    _to_torch_tensor,
    _update_atomic_numbers_from_configurations,
)


__all__ = [
    "create_pdb_from_xyz",
    "load_traj_with_ase",
]

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
    trajectories : list[ase.Atoms] or list[list[ase.Atoms]]
        ASE trajectory objects to convert.
    graph_labels : list, optional
        Frame-level graph labels for each trajectory.
    node_labels : list, optional
        Node-level labels for each trajectory.
    system_selection : optional
        Atom selection defining the system atoms.
    environment_selection : optional
        Atom selection defining the environment atoms.
    subsystem_selection : optional
        Atom selection defining the subsystem atoms.
    lengths_conversion : float, optional
        Length conversion factor. ASE coordinates are already in Angstroms,
        so the default is 1.0.
    atom_names : list, optional
        Names of system atoms. If not provided, they are inferred from
        the ASE Atoms objects.

    Returns
    -------
    configurations : Configurations
        Atomic configurations generated from all trajectory frames.
    atomic_numbers : AtomicNumberTable
        Atomic number table containing all species.
    atom_names : list
        Names of the selected system atoms.
    """
    if not isinstance(trajectories, list):
        raise TypeError(
            "`trajectories` must be a list of ase.Atoms "
            "or a list of trajectory lists."
        )

    if not trajectories:
        raise ValueError("`trajectories` cannot be empty.")

    if isinstance(trajectories[0], ase.Atoms):
        trajectories = [trajectories]

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
        configuration = _configurations_from_ase_trajectory(
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
        atom_names = _names_from_ase_atoms(
            ase_atoms_list=trajectories,
            system_selection=system_selection,
        )

    return configurations, atomic_numbers, atom_names


def load_traj_with_ase(
    trajectory: str,
    start: int = 0,
    stop: int = None,
    stride: int = 1,
) -> List[ase.Atoms]:
    """Load a trajectory using ASE.

    Parameters
    ----------
    trajectory : str
        Path to the trajectory file.
    start : int, optional
        Starting frame index, by default 0.
    stop : int, optional
        Stopping frame index, by default None.
    stride : int, optional
        Stride for frame selection, by default 1.

    Returns
    -------
    List[ase.Atoms]
        Loaded ASE trajectory frames.
    """
    if stop is None:
        stop = ""

    frame_selection = f"{start}:{stop}:{stride}"
    return read(trajectory, index=frame_selection)


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


def _configurations_from_ase_trajectory(
    trajectory: List[ase.Atoms],
    graph_labels: list = None,
    node_labels: list = None,
    system_selection: Any = None,
    environment_selection: Any = None,
    subsystem_selection: Any = None,
    lengths_conversion: float = 1.0,
) -> Configurations:
    """Create configurations from one ASE trajectory."""

    if isinstance(trajectory, Atoms):
        trajectory = [trajectory]

    selected_atoms = {
        "system": list(range(len(trajectory[0]))),
        "environment": [],
        "subsystem": None,
    }

    selections = {
        "system": system_selection,
        "environment": environment_selection,
        "subsystem": subsystem_selection,
    }

    for name, selection in selections.items():
        if selection is None:
            continue

        selected_atoms[name] = _selection_to_indices(
            selection,
            trajectory[0],
        )

        if not selected_atoms[name]:
            raise ValueError(
                f"No atoms will be selected with "
                f"{name}_selection: {selection}!"
            )

    if subsystem_selection is not None:
        if not set(selected_atoms["subsystem"]).issubset(
            selected_atoms["system"]
        ):
            raise ValueError(
                "Only atoms in `system_selection` can be selected "
                "by `subsystem_selection`!"
            )

    system_indices = selected_atoms["system"]
    environment_indices = selected_atoms["environment"]
    subsystem_indices = selected_atoms["subsystem"]

    required_indices = system_indices + environment_indices

    sliced_trajectory = [
        frame[required_indices]
        for frame in trajectory
    ]

    system_index_map = {
        original: new
        for new, original in enumerate(system_indices)
    }

    selected_atoms["system"] = list(
        range(len(system_indices))
    )

    selected_atoms["environment"] = list(
        range(
            len(system_indices),
            len(system_indices) + len(environment_indices),
        )
    )

    if subsystem_indices is not None:
        selected_atoms["subsystem"] = [
            system_index_map[index]
            for index in subsystem_indices
        ]

    configurations = []

    for i, frame in enumerate(sliced_trajectory):
        label_i = (
            _to_torch_tensor(graph_labels[i]).reshape(-1, 1)
            if graph_labels is not None
            else None
        )

        node_i = (
            _to_torch_tensor(node_labels[i]).reshape(-1, 1)
            if node_labels is not None
            else None
        )

        configurations.append(
            Configuration(
                atomic_numbers=frame.get_atomic_numbers().tolist(),
                positions=frame.get_positions() * lengths_conversion,
                cell=frame.get_cell().array * lengths_conversion,
                pbc=frame.get_pbc().tolist(),
                graph_labels=label_i,
                node_labels=node_i,
                system=selected_atoms["system"],
                environment=selected_atoms["environment"],
                subsystem=selected_atoms["subsystem"],
            )
        )

    return configurations


def _names_from_ase_atoms(
    ase_atoms_list,
    system_selection: Any,
) -> List[str]:
    """Return atom names for the selected system atoms."""

    first = ase_atoms_list[0]

    if isinstance(first, (list, tuple)):
        first = first[0]

    indices = _selection_to_indices(
        system_selection,
        first,
    )

    return first[indices].get_chemical_symbols()


def _ase_from_graphs(
    graphs,
    atomic_numbers,
) -> List[Atoms]:
    """Convert mlcolvar graphs to ASE Atoms."""

    atomic_numbers = np.asarray(
        atomic_numbers,
        dtype=int,
    )

    atoms_list = []

    for graph in graphs:
        species = (
            graph["node_attrs"]
            .argmax(dim=-1)
            .cpu()
            .numpy()
        )

        atoms_list.append(
            Atoms(
                numbers=atomic_numbers[species],
                positions=graph["positions"].cpu().numpy(),
                cell=graph["cell"].cpu().numpy(),
                pbc=graph["pbc"].cpu().numpy().reshape(-1),
            )
        )

    return atoms_list


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