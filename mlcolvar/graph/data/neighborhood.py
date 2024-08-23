import numpy as np
from matscipy.neighbours import neighbour_list
from typing import Optional, Tuple, List

"""
The neighbor list function. This module is taken from MACE directly:
https://github.com/ACEsuit/mace/blob/main/mace/data/neighborhood.py
"""

__all__ = ['get_neighborhood']


def get_neighborhood(
    positions: np.ndarray,  # [num_positions, 3]
    cutoff: float,
    pbc: Optional[Tuple[bool, bool, bool]] = None,
    cell: Optional[np.ndarray] = None,  # [3, 3]
    true_self_interaction: Optional[bool] = False,
    system_indices: Optional[List[int]] = None,
    environment_indices: Optional[List[int]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Get the neighbor list of a given set atoms.

    Parameters
    ----------
    positions: numpy.ndarray (shape: [N, 3])
        The positions array.
    cutoff: float
        The cutoff radius.
    pbc: Tuple[bool, bool, bool] (shape: [3])
        If enable PBC in the directions of the three lattice vectors.
    cell: numpy.ndarray (shape: [3, 3])
        The lattice vectors.
    true_self_interaction: bool
        If keep self-edges that don't cross periodic boundaries.
    system_indices: List[int]
        Indices of the system atoms.
    environment_indices: List[int]
        Indices of the environment atoms.

    Returns
    -------
    edge_index: numpy.ndarray (shape: [2, n_edges])
        The edge indices in the graph.
    shifts: numpy.ndarray (shape: [n_edges, 3])
        The shift vectors (unit_shifts * cell_lengths).
    unit_shifts: numpy.ndarray (shape: [n_edges, 3])
        The unit shift vectors (number of PBC croessed by the edges).

    Notes
    -----
    Arguments `system_indices` and `environment_indices` must presnet at the
    same time. When these arguments are given, only edges in the [subsystem]
    formed by [the systems atoms] and [the environment atoms within the cutoff
    radius of the systems atoms] will be kept.
    Besides, these two lists could not contain common atoms.
    """

    if system_indices is not None or environment_indices is not None:
        assert system_indices is not None and environment_indices is not None

        system_indices = np.array(system_indices)
        environment_indices = np.array(environment_indices)
        assert np.intersect1d(system_indices, environment_indices).size == 0

    if pbc is None:
        pbc = (False, False, False)

    if cell is None or cell.any() == np.zeros((3, 3)).any():
        cell = np.identity(3, dtype=float)

    assert len(pbc) == 3 and all(isinstance(i, (bool, np.bool_)) for i in pbc)
    assert cell.shape == (3, 3)

    pbc_x = pbc[0]
    pbc_y = pbc[1]
    pbc_z = pbc[2]
    identity = np.identity(3, dtype=float)
    max_positions = np.max(np.absolute(positions)) + 1
    # Extend cell in non-periodic directions
    # For models with more than 5 layers, the multiplicative constant needs to
    # be increased.
    if not pbc_x:
        cell[:, 0] = max_positions * 5 * cutoff * identity[:, 0]
    if not pbc_y:
        cell[:, 1] = max_positions * 5 * cutoff * identity[:, 1]
    if not pbc_z:
        cell[:, 2] = max_positions * 5 * cutoff * identity[:, 2]

    sender, receiver, unit_shifts = neighbour_list(
        quantities='ijS',
        pbc=pbc,
        cell=cell,
        positions=positions,
        cutoff=float(cutoff),
        # self_interaction=True,  # we want edges from atom to itself in different periodic images
        # use_scaled_positions=False,  # positions are not scaled positions
    )

    if not true_self_interaction:
        # Eliminate self-edges that don't cross periodic boundaries
        true_self_edge = sender == receiver
        true_self_edge &= np.all(unit_shifts == 0, axis=1)
        keep_edge = ~true_self_edge

        # NOTE: after eliminating self-edges, it can be that no edges remain
        # in this system
        sender = sender[keep_edge]
        receiver = receiver[keep_edge]
        unit_shifts = unit_shifts[keep_edge]

    if system_indices is not None:
        # Get environment atoms that are neighbors of the system.
        keep_edge = np.where(np.in1d(receiver, system_indices))[0]
        keep_sender = np.intersect1d(sender[keep_edge], environment_indices)
        keep_atom = np.concatenate((system_indices, np.unique(keep_sender)))
        # Get the edges in the subsystem.
        keep_sender = np.where(np.in1d(sender, keep_atom))[0]
        keep_receiver = np.where(np.in1d(receiver, keep_atom))[0]
        keep_edge = np.intersect1d(keep_sender, keep_receiver)
        # Get the edges
        sender = sender[keep_edge]
        receiver = receiver[keep_edge]
        unit_shifts = unit_shifts[keep_edge]

    # Build output
    edge_index = np.stack((sender, receiver))  # [2, n_edges]

    # From the docs: With the shift vector S, the distances D between atoms
    # can be computed from: D = positions[j]-positions[i]+S.dot(cell)
    shifts = np.dot(unit_shifts, cell)  # [n_edges, 3]

    return edge_index, shifts, unit_shifts


def test_get_neighborhood() -> None:

    positions = np.array(
        [[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3]], dtype=float
    )
    cell = np.array([[4, 0, 0], [0, 4, 0], [0, 0, 4]], dtype=float)

    n, s, u = get_neighborhood(positions, cutoff=5.0)
    assert (
        n == np.array(
            [[0, 0, 1, 1, 1, 2, 2, 2, 3, 3], [1, 2, 0, 2, 3, 0, 1, 3, 1, 2]],
            dtype=int
        )
    ).all()

    n, s, u = get_neighborhood(positions, cutoff=2.0)
    assert (
        n == np.array([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]], dtype=int)
    ).all()

    n, s, u = get_neighborhood(
        positions, cutoff=2.0, pbc=[True] * 3, cell=cell
    )
    assert (
        n == np.array(
            [[0, 0, 1, 1, 2, 2, 3, 3], [3, 1, 0, 2, 1, 3, 2, 0]], dtype=int
        )
    ).all()
    assert (
        s == np.array(
            [
                [-4.0, -4.0, -4.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [4.0, 4.0, 4.0]
            ],
            dtype=float
        )
    ).all()
    assert (
        u == np.array(
            [
                [-1, -1, -1],
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
                [1, 1, 1]
            ],
            dtype=int
        )
    ).all()

    n, s, u = get_neighborhood(
        positions,
        cutoff=2.0,
        pbc=[True] * 3,
        cell=cell,
        system_indices=[0, 1],
        environment_indices=[2, 3]
    )
    assert (
        n == np.array(
            [[0, 0, 1, 1, 2, 2, 3, 3], [3, 1, 0, 2, 1, 3, 2, 0]], dtype=int
        )
    ).all()

    n, s, u = get_neighborhood(
        positions,
        cutoff=2.0,
        pbc=[True] * 3,
        cell=cell,
        system_indices=[0],
        environment_indices=[1, 2, 3]
    )
    assert (
        n == np.array(
            [[0, 0, 1, 3], [3, 1, 0, 0]], dtype=int
        )
    ).all()
    assert (
        s == np.array(
            [
                [-4.0, -4.0, -4.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [4.0, 4.0, 4.0]
            ],
            dtype=float
        )
    ).all()
    assert (
        u == np.array(
            [[-1, -1, -1], [0, 0, 0], [0, 0, 0], [1, 1, 1]],
            dtype=int
        )
    ).all()


if __name__ == "__main__":
    test_get_neighborhood()
