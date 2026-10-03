import pytest
import torch

from mlcolvar.representation import SelectAtoms


def test_select_atoms():
    features = torch.tensor(
        [
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
            [7.0, 8.0],
            [9.0, 10.0],
            [11.0, 12.0],
        ]
    )
    ptr = torch.tensor([0, 3, 6])

    output = SelectAtoms([0, 2])(features, ptr)

    expected = torch.tensor(
        [
            [1.0, 2.0, 5.0, 6.0],
            [7.0, 8.0, 11.0, 12.0],
        ]
    )
    torch.testing.assert_close(output, expected)


@pytest.mark.parametrize(
    "atom_indices",
    [
        [],
        [-1],
        [0, 0],
    ],
)
def test_select_atoms_invalid_indices(atom_indices):
    with pytest.raises(ValueError):
        SelectAtoms(atom_indices)


def test_select_atoms_out_of_range():
    features = torch.ones(3, 2)
    ptr = torch.tensor([0, 2, 3])

    with pytest.raises(RuntimeError):
        SelectAtoms([1])(features, ptr)