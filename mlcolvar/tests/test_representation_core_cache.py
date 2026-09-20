from typing import Dict, Optional

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
from mlcolvar.representation.cache import CachedRepresentationDerivatives


class DummyVectorRepresentation(VectorRepresentation):
    def __init__(self, freeze=True):
        super().__init__(
            in_features=3,
            out_features=2,
            freeze=freeze,
        )
        self.encoder = nn.Linear(3, 2, bias=False)
        self.encoder.weight.data.copy_(
            torch.tensor([
                [1.0, 0.0, 1.0],
                [0.0, 1.0, -1.0],
            ])
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
        self.scale = nn.Parameter(
            torch.tensor(2.0),
            requires_grad=False,
        )

    def forward(self, data, cell=None):
        return data["atom_features"] * self.scale


class IdentityDescriptorDerivatives(SmartDerivatives):
    def __init__(self):
        nn.Module.__init__(self)

    def forward(self, x, ref_idx=None):
        return x


def make_dataset():
    return DictDataset({
        "data": torch.randn(4, 3)
    })


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
        "node_attrs": torch.tensor(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 0.0],
                [0.0, 1.0],
            ]
        ),
        "batch": torch.tensor([0, 0, 1, 1]),
        "ptr": torch.tensor([0, 2, 4]),
        "edge_index": torch.empty((2, 0), dtype=torch.long),
        "shifts": torch.empty((0, 3)),
    }


def test_vector_representation_model():
    representation = DummyVectorRepresentation()
    model = RepresentationModel(
        representation,
        n_out=1,
        hidden_layers=(),
    )

    x = torch.randn(2, 3, requires_grad=True)
    output = model(x)

    assert output.shape == (2, 1)
    assert all(
        not p.requires_grad
        for p in representation.parameters()
    )

    output.sum().backward()
    assert x.grad is not None


@pytest.mark.parametrize(
    ("transform", "out_features"),
    [
        (lambda x: x.pool("mean"), 2),
        (lambda x: x.concat_atoms([0, 1]), 4),
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

    assert model(make_graph()).shape == (2, 1)


def test_representation_cache():
    representation = DummyVectorRepresentation()
    dataset = make_dataset()

    cache = representation.cache(
        dataset,
        batch_size=2,
    )

    assert cache.features.shape == (4, 2)
    assert cache.jacobian is None
    assert cache.reference_indices is None


def test_model_cache():
    representation = DummyVectorRepresentation()
    model = RepresentationModel(
        representation,
        n_out=1,
        hidden_layers=(),
    )
    dataset = make_dataset()

    cache = model.cache(dataset)

    with torch.no_grad():
        expected = representation(
            dataset["data"]
        )

    torch.testing.assert_close(
        cache.features,
        expected,
    )


def test_selective_jacobian_cache():
    cache = DummyVectorRepresentation().cache(
        make_dataset(),
        jacobian=True,
        descriptor_derivatives=IdentityDescriptorDerivatives(),
        jacobian_indices=torch.tensor([1, 3]),
        batch_size=1,
    )

    assert cache.jacobian.shape == (2, 3, 2)
    torch.testing.assert_close(
        cache.reference_indices,
        torch.tensor([-1, 0, -1, 1]),
    )


def test_cache_requires_frozen_representation():
    with pytest.raises(
        RuntimeError,
        match="frozen representation",
    ):
        DummyVectorRepresentation(
            freeze=False
        ).cache(
            make_dataset()
        )


def test_cached_derivatives_require_ref_idx():
    derivatives = CachedRepresentationDerivatives(
        torch.zeros(2, 3, 2)
    )

    with pytest.raises(
        ValueError,
        match="ref_idx",
    ):
        derivatives(
            torch.zeros(2, 2)
        )