from typing import Dict

import pytest
import torch
from torch import nn

from mlcolvar.core.loss.utils.smart_derivatives import SmartDerivatives
from mlcolvar.data import DictDataset
from mlcolvar.representation import (
    GraphRepresentation,
    RepresentationModel,
    VectorRepresentation,
)
from mlcolvar.representation.cache import (
    CachedRepresentationDerivatives,
)


class DummyVectorRepresentation(VectorRepresentation):
    def __init__(self):
        super().__init__(
            in_features=3,
            out_features=2,
            freeze=True,
        )

        self.encoder = nn.Linear(
            3,
            2,
            bias=False,
        )

        self._freeze_module(self.encoder)

    def forward(self, x, cell=None):
        return self.encoder(x)


class DummyAtomRepresentation(GraphRepresentation):
    def __init__(self):
        super().__init__(
            out_features=2,
            atomic_numbers=[1, 8],
            cutoff=5.0,
            output_kind="atom",
            freeze=True,
        )

    def forward(self, data, cell=None):
        return data["atom_features"]


class IdentityDescriptorDerivatives(SmartDerivatives):
    def __init__(self):
        nn.Module.__init__(self)

    def forward(self, x, ref_idx=None):
        return x


def make_dataset():
    return DictDataset(
        {
            "data": torch.randn(4, 3),
        }
    )


def make_graph() -> Dict[str, torch.Tensor]:
    return {
        "atom_features": torch.tensor(
            [
                [1.0, 2.0],
                [3.0, 4.0],
                [2.0, 4.0],
                [4.0, 6.0],
            ]
        ),
        "positions": torch.zeros(4, 3),
        "batch": torch.tensor([0, 0, 1, 1]),
        "ptr": torch.tensor([0, 2, 4]),
    }


def test_vector_representation_model():
    representation = DummyVectorRepresentation()

    model = RepresentationModel(
        representation,
        n_out=1,
        hidden_layers=(),
    )

    x = torch.randn(
        2,
        3,
        requires_grad=True,
    )

    output = model(x)

    assert output.shape == (2, 1)

    output.sum().backward()

    assert x.grad is not None

    assert all(
        not parameter.requires_grad
        for parameter in representation.parameters()
    )


@pytest.mark.parametrize(
    ("transform", "out_features"),
    [
        (lambda x: x.pool("mean"), 2),
        (
            lambda x: x.concat_atoms([0, 1]),
            4,
        ),
    ],
)
def test_graph_representation_transforms(
    transform,
    out_features,
):
    representation = transform(
        DummyAtomRepresentation()
    )

    assert representation.output_kind == "system"
    assert representation.out_features == out_features

    model = RepresentationModel(
        representation,
        n_out=1,
        hidden_layers=(),
    )

    assert model(
        make_graph()
    ).shape == (2, 1)


def test_representation_cache():
    representation = DummyVectorRepresentation()
    dataset = make_dataset()

    cache = representation.cache(
        dataset,
        batch_size=2,
    )

    with torch.no_grad():
        expected = representation(
            dataset["data"]
        )

    torch.testing.assert_close(
        cache.features,
        expected,
    )

    assert cache.jacobian is None


def test_jacobian_cache_and_derivatives():
    cache = DummyVectorRepresentation().cache(
        make_dataset(),
        jacobian=True,
        descriptor_derivatives=(
            IdentityDescriptorDerivatives()
        ),
        jacobian_indices=torch.tensor(
            [1, 3]
        ),
        batch_size=1,
    )

    assert cache.jacobian.shape == (
        2,
        3,
        2,
    )

    torch.testing.assert_close(
        cache.reference_indices,
        torch.tensor(
            [-1, 0, -1, 1]
        ),
    )

    derivatives = CachedRepresentationDerivatives(
        cache.jacobian
    )

    output = derivatives(
        torch.ones(2, 2),
        torch.tensor([0, 1]),
    )

    assert output.shape == (2, 3)