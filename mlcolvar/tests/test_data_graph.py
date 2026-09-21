import numpy as np

from mlcolvar.data.graph.atomic import AtomicNumberTable
from mlcolvar.data.graph.neighborhood import get_neighborhood


def test_atomic_number_table() -> None:
    table = AtomicNumberTable([1, 6, 7, 8])

    numbers = np.array([1, 7, 6, 8])
    assert (
        table.zs_to_indices(numbers) == np.array([0, 2, 1, 3], dtype=int)
    ).all()

    numbers = np.array([1, 1, 1, 6, 8, 1])
    assert (
        table.zs_to_indices(numbers) == np.array([0, 0, 0, 1, 3, 0], dtype=int)
    ).all()

    table_1 = AtomicNumberTable.from_zs([6] * 3 + [1] * 10 + [7] * 3 + [8] * 2)
    assert table_1.zs == table.zs
    

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