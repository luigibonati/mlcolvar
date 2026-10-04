import pytest
import torch

from mlcolvar.core import FeedForward
from mlcolvar.cvs import RegressionCV
from mlcolvar.representation import Representation, SelectAtoms


class AtomRepresentation(Representation):
    def __init__(self, pooling_operation=None):
        super().__init__(
            out_features=2,
            input_kind="graph",
            output_kind="atom" if pooling_operation is None else "system",
            atomic_numbers=[1],
            cutoff=5.0,
            pooling_operation=pooling_operation,
        )

    def forward(self, data, cell=None):
        return self.pooling(data["features"], data)


def graph():
    return {
        "features": torch.tensor([
            [1., 2.], [3., 4.], [5., 6.],
            [7., 8.], [9., 10.], [11., 12.],
        ]),
        "batch": torch.tensor([0, 0, 0, 1, 1, 1]),
        "ptr": torch.tensor([0, 3, 6]),
    }


def test_select_atoms():
    representation = SelectAtoms(AtomRepresentation(), [0, 2])
    expected = torch.tensor([[1., 2., 5., 6.], [7., 8., 11., 12.]])

    torch.testing.assert_close(representation(graph()), expected)

    assert representation.input_kind == "graph"
    assert representation.output_kind == "system"
    assert representation.out_features == 4

    model = RegressionCV(
        model=FeedForward([representation.out_features, 1]),
        preprocessing=representation,
    )
    assert model(graph()).shape == (2, 1)


def test_select_atoms_errors():
    for indices in ([], [-1], [0, 0]):
        with pytest.raises(ValueError):
            SelectAtoms(AtomRepresentation(), indices)

    with pytest.raises(ValueError):
        SelectAtoms(AtomRepresentation("mean"), [0])

    with pytest.raises(RuntimeError):
        SelectAtoms(AtomRepresentation(), [3])(graph())