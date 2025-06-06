import warnings
import numpy as np
import mdtraj as md
from dataclasses import dataclass
from typing import List, Iterable, Optional

"""
The helper functions for atomic data. This module is taken from MACE directly:
https://github.com/ACEsuit/mace/blob/main/mace/tools/utils.py
https://github.com/ACEsuit/mace/blob/main/mace/data/utils.py
"""

__all__ = ['AtomicNumberTable', 'Configuration', 'Configurations']


class AtomicNumberTable:
    """The atomic number table. 
    Used to map between one hot encodings and a given set of actual atomic numbers.
    """

    def __init__(self, zs: List[int]) -> None:
        """Initializes an atomi number table object

        Parameters
        ----------
        zs: List[int]
            The atomic numbers in this table
        """
        self.zs = zs
        self.masses = [1.0] * len(zs)
        for i in range(len(zs)):
            try:
                m = md.element.Element.getByAtomicNumber(zs[i]).mass
                self.masses[i] = m
            except Exception:
                warnings.warn(
                    'Can not assign mass for atom number: {:d}'.format(zs[i])
                )

    def __len__(self) -> int:
        """Number of elements in the table"""
        return len(self.zs)

    def __str__(self) -> str:
        return f'AtomicNumberTable: {tuple(s for s in self.zs)}'

    def index_to_z(self, index: int) -> int:
        """Maps the encoding to the actual atomic number
        
        Parameters
        ----------
        index: int
            Index of the encoding to be mapped
        """
        return self.zs[index]

    def index_to_symbol(self, index: int) -> str:
        """Map the encoding to the atomic symbol

        Parameters
        ----------
        index: int
            Index of the encoding to be mapped
        """
        return md.element.Element.getByAtomicNumber(self.zs[index]).symbol

    def z_to_index(self, atomic_number: int) -> int:
        """Maps an atomic number to the encoding.

        Parameters
        ----------
        atomic_number: int
            The atomic number to be mapped
        """
        return self.zs.index(atomic_number)

    def zs_to_indices(self, atomic_numbers: np.ndarray) -> np.ndarray:
        """Maps an array of atomic number to the encodings.
        
        Parameters
        ----------
        atomic_numbers: numpy.ndarray
            The atomic numbers to be mapped
        """
        to_index_fn = np.vectorize(self.z_to_index)
        return to_index_fn(atomic_numbers)

    @classmethod
    def from_zs(cls, atomic_numbers: Iterable[int]) -> 'AtomicNumberTable':
        """Build the table from an array atomic numbers.

        Parameters
        ----------
        atomic_numbers: Iterable[int]
            The atomic numbers to be used for building the table
        """
        z_set = set()
        for z in atomic_numbers:
            z_set.add(z)
        return cls(sorted(list(z_set)))


def get_masses(atomic_numbers: Iterable[int]) -> List[float]:
    """Get atomic masses from atomic numbers.

    Parameters
    ----------
    atomic_numbers: Iterable[int]
        The atomic numbers for which to return the atomic masses
    """
    return AtomicNumberTable.from_zs(atomic_numbers).masses.copy()


@dataclass
class Configuration:
    """
    Internal helper class that describe a given configuration of the system.
    """
    atomic_numbers: np.ndarray          # shape: [n_atoms]
    positions: np.ndarray               # shape: [n_atoms, 3], units: Ang
    cell: np.ndarray                    # shape: [n_atoms, 3], units: Ang
    pbc: Optional[tuple]                # shape: [3]
    node_labels: Optional[np.ndarray]   # shape: [n_atoms, n_node_labels]
    graph_labels: Optional[np.ndarray]  # shape: [n_graph_labels, 1]
    weight: Optional[float] = 1.0       # shape: []
    system: Optional[np.ndarray] = None       # shape: [n_system_atoms]
    environment: Optional[np.ndarray] = None  # shape: [n_environment_atoms]


Configurations = List[Configuration]


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


if __name__ == '__main__':
    test_atomic_number_table()
