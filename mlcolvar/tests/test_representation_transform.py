import pytest
import torch

from mlcolvar.representation import SelectAtoms


def test_select_atoms():
    features = torch.tensor([
        [1., 2.], [3., 4.], [5., 6.],
        [7., 8.], [9., 10.], [11., 12.],
    ])
    output = SelectAtoms([0, 2])(features, torch.tensor([0, 3, 6]))
    expected = torch.tensor([[1., 2., 5., 6.], [7., 8., 11., 12.]])
    torch.testing.assert_close(output, expected)


@pytest.mark.parametrize("indices", [[], [-1], [0, 0]])
def test_select_atoms_errors(indices):
    with pytest.raises(ValueError):
        SelectAtoms(indices)


def test_select_atoms_out_of_range():
    with pytest.raises(RuntimeError):
        SelectAtoms([1])(
            torch.ones(3, 2),
            torch.tensor([0, 2, 3]),
        )