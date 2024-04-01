import numpy as np
from matscipy.neighbours import neighbour_list
from typing import Optional, Tuple, List

"""
The neighbour list function. This module is taken from MACE directly:
https://github.com/ACEsuit/mace/blob/main/mace/data/neighborhood.py
"""

__all__ = ['get_neighborhood']


def get_neighborhood(
    positions: np.ndarray,  # [num_positions, 3]
    cutoff: float,
    pbc: Optional[Tuple[bool, bool, bool]] = None,
    cell: Optional[np.ndarray] = None,  # [3, 3]
    true_self_interaction: Optional[bool] = False,
    sender_indices: Optional[List[int]] = None,
    receiver_indices: Optional[List[int]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Get the neighbour list of a given set atoms.

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
    sender_indices: List[int]
        Indices of senders. If given, only edges sent by these atoms will be
        kept in the graph.
    receiver_indices: List[int]
        Indices of receiver. If given, only edges received by these atoms will
        be kept in the graph.

    Returns
    -------
    edge_index: numpy.ndarray (shape: [2, n_edges])
        The edge indices in the graph.
    shifts: numpy.ndarray (shape: [n_edges, 3])
        The shift vectors (unit_shifts * cell_lengths).
    unit_shifts: numpy.ndarray (shape: [n_edges, 3])
        The unit shift vectors (number of PBC croessed by the edges).
    """
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
        quantities="ijS",
        pbc=pbc,
        cell=cell,
        positions=positions,
        cutoff=cutoff,
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

    if sender_indices is not None:
        sender_indices = np.array(sender_indices, dtype=int)
        keep_edge = np.where(np.in1d(sender, sender_indices))[0]
        sender = sender[keep_edge]
        receiver = receiver[keep_edge]
        unit_shifts = unit_shifts[keep_edge]

    if receiver_indices is not None:
        receiver_indices = np.array(receiver_indices, dtype=int)
        keep_edge = np.where(np.in1d(receiver, receiver_indices))[0]
        sender = sender[keep_edge]
        receiver = receiver[keep_edge]
        unit_shifts = unit_shifts[keep_edge]

    # Build output
    edge_index = np.stack((sender, receiver))  # [2, n_edges]

    # From the docs: With the shift vector S, the distances D between atoms
    # can be computed from: D = positions[j]-positions[i]+S.dot(cell)
    shifts = np.dot(unit_shifts, cell)  # [n_edges, 3]

    return edge_index, shifts, unit_shifts


def test_get_neighborhood():

    positions = np.array(
        [[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3]], dtype=float
    )
    cell = np.array([[2, 0, 0], [0, 2, 0], [0, 0, 2]], dtype=float)

    n, s, u = get_neighborhood(positions, cutoff=5.0)
    assert (
        n == np.array(
            [[0, 0, 1, 1, 1, 2, 2, 2, 3, 3], [1, 2, 0, 2, 3, 0, 1, 3, 1, 2]],
            dtype=int
        )
    ).all()

    n, s, u = get_neighborhood(positions, cutoff=5.0, receiver_indices=[0, 1])
    assert (
        n == np.array([[0, 1, 2, 2, 3], [1, 0, 0, 1, 1]], dtype=int)
    ).all()

    n, s, u = get_neighborhood(positions, cutoff=5.0, sender_indices=[0, 1])
    assert (
        n == np.array([[0, 0, 1, 1, 1], [1, 2, 0, 2, 3]], dtype=int)
    ).all()

    n, s, u = get_neighborhood(positions, cutoff=2.0)
    assert (
        n == np.array([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]], dtype=int)
    ).all()

    n, s, u = get_neighborhood(
        positions, cutoff=1.0, pbc=[True] * 3, cell=cell
    )
    assert (
        n == np.array([[0, 1, 2, 3], [2, 3, 0, 1]], dtype=int)
    ).all()
    assert (
        s == np.array(
            [[-2, -2, -2], [-2, -2, -2], [2, 2, 2], [2, 2, 2]],
            dtype=float
        )
    ).all()
    assert (
        u == np.array(
            [[-1, -1, -1], [-1, -1, -1], [1, 1, 1], [1, 1, 1]],
            dtype=int
        )
    ).all()

    n, s, u = get_neighborhood(
        positions, cutoff=1.0, pbc=[True] * 3, cell=cell, receiver_indices=[0]
    )
    assert (n == np.array([[2], [0]], dtype=int)).all()
    assert (s == np.array([[2, 2, 2]], dtype=float)).all()
    assert (u == np.array([[1, 1, 1]], dtype=int)).all()

    n, s, u = get_neighborhood(
        positions, cutoff=1.0, pbc=[True] * 3, cell=cell, sender_indices=[0]
    )
    assert (n == np.array([[0], [2]], dtype=int)).all()
    assert (s == np.array([[-2, -2, -2]], dtype=float)).all()
    assert (u == np.array([[-1, -1, -1]], dtype=int)).all()


if __name__ == "__main__":
    test_get_neighborhood()
