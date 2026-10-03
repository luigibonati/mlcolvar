from typing import Optional

import pytest
import torch
from torch import nn

from mlcolvar.core import FeedForward
from mlcolvar.cvs import RegressionCV
from mlcolvar.data import DictDataset
from mlcolvar.representation import Representation, evaluate_dataset
from mlcolvar.representation.preparation import compute_jacobian


class DummyVectorRepresentation(Representation):
    def __init__(self):
        super().__init__(
            in_features=3,
            out_features=2,
            input_kind="vector",
            output_kind="system",
            freeze=True,
        )
        self.encoder = nn.Linear(3, 2, bias=False)
        self._freeze_module(self.encoder)

    def forward(self, x, cell=None):
        return self.encoder(x)


class DummyAtomRepresentation(Representation):
    def __init__(
        self,
        pooling_operation: Optional[str] = None,
    ):
        super().__init__(
            out_features=2,
            input_kind="graph",
            atomic_numbers=[1, 8],
            cutoff=5.0,
            pooling_operation=pooling_operation,
            freeze=True,
        )

    def forward(self, data, cell=None):
        return self.pooling(data["atom_features"], data)


def make_dataset():
    return DictDataset({
        "data": torch.randn(4, 3),
    })


def make_graph() -> dict[str, torch.Tensor]:
    return {
        "atom_features": torch.tensor([
            [1.0, 2.0],
            [3.0, 4.0],
            [2.0, 4.0],
            [4.0, 6.0],
        ]),
        "positions": torch.zeros(4, 3),
        "batch": torch.tensor([0, 0, 1, 1]),
        "ptr": torch.tensor([0, 2, 4]),
    }


def test_vector_representation_preprocessing():
    representation = DummyVectorRepresentation()
    model = RegressionCV(
        model=FeedForward([representation.out_features, 1]),
        preprocessing=representation,
    )

    x = torch.randn(2, 3, requires_grad=True)
    output = model(x)
    assert output.shape == (2, 1)

    output.sum().backward()
    assert x.grad is not None
    assert all(
        not parameter.requires_grad
        for parameter in representation.parameters()
    )


@pytest.mark.parametrize(
    ("factory", "out_features"),
    [
        (
            lambda: DummyAtomRepresentation(pooling_operation="mean"),
            2,
        ),
        (
            lambda: DummyAtomRepresentation().concat_atoms([0, 1]),
            4,
        ),
    ],
)
def test_graph_representation_transforms(factory, out_features):
    representation = factory()

    assert representation.input_kind == "graph"
    assert representation.output_kind == "system"
    assert representation.out_features == out_features

    model = RegressionCV(
        model=FeedForward([out_features, 1]),
        preprocessing=representation,
    )

    assert model(make_graph()).shape == (2, 1)


def test_evaluate_dataset():
    model = nn.Linear(3, 2, bias=False)
    dataset = make_dataset()

    output = evaluate_dataset(
        model,
        dataset,
        batch_size=2,
    )

    with torch.no_grad():
        expected = model(dataset["data"])

    torch.testing.assert_close(output, expected)


def test_compute_jacobian():
    model = nn.Linear(3, 2, bias=False)
    dataset = make_dataset()

    jacobian = compute_jacobian(
        model,
        dataset,
        indices=torch.tensor([1, 3]),
        batch_size=1,
    )

    assert jacobian.shape == (2, 3, 2)

    expected = (
        model.weight.detach()
        .T.unsqueeze(0)
        .expand(2, -1, -1)
    )

    torch.testing.assert_close(jacobian, expected)