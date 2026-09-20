from __future__ import annotations

from typing import Dict, Optional

import pytest
import torch
from torch import nn

from mlcolvar.core import BaseGNN
from mlcolvar.core.loss.utils.smart_derivatives import SmartDerivatives
from mlcolvar.data import DictDataset
from mlcolvar.representation import (
    GraphRepresentation,
    RepresentationModel,
    TaskHead,
    VectorRepresentation,
)
from mlcolvar.representation.cache import (
    CachedRepresentationDerivatives,
)


class DummyVectorRepresentation(VectorRepresentation):
    def __init__(self, freeze: bool = True) -> None:
        super().__init__(
            in_features=3,
            out_features=2,
            output_kind="system",
            freeze=freeze,
        )

        self.encoder = nn.Linear(
            3,
            2,
            bias=False,
        )

        with torch.no_grad():
            self.encoder.weight.copy_(
                torch.tensor(
                    [
                        [1.0, 0.0, 1.0],
                        [0.0, 1.0, -1.0],
                    ]
                )
            )

        self._freeze_module(
            self.encoder
        )

    def forward(
        self,
        x: torch.Tensor,
        cell: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        del cell
        return self.encoder(x)


class DummyAtomRepresentation(GraphRepresentation):
    def __init__(self) -> None:
        super().__init__(
            out_features=2,
            atomic_numbers=[1, 8],
            cutoff=5.0,
            output_kind="atom",
            freeze=True,
        )

        self.scale = nn.Parameter(
            torch.tensor(2.0)
        )
        self.scale.requires_grad_(False)

    def forward(
        self,
        data: Dict[str, torch.Tensor],
        cell: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        del cell
        return (
            data["atom_features"]
            * self.scale
        )


class IdentityDescriptorDerivatives(SmartDerivatives):
    def __init__(self) -> None:
        nn.Module.__init__(self)

    def forward(
        self,
        x: torch.Tensor,
        ref_idx: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        del ref_idx
        return x


def make_graph() -> Dict[str, torch.Tensor]:
    return {
        "atom_features": torch.tensor(
            [
                [1.0, 2.0],
                [3.0, 4.0],
                [2.0, 4.0],
                [4.0, 6.0],
            ],
            requires_grad=True,
        ),
        "positions": torch.zeros(
            4,
            3,
        ),
        "node_attrs": torch.tensor(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 0.0],
                [0.0, 1.0],
            ]
        ),
        "batch": torch.tensor(
            [0, 0, 1, 1],
            dtype=torch.long,
        ),
        "ptr": torch.tensor(
            [0, 2, 4],
            dtype=torch.long,
        ),
        "edge_index": torch.empty(
            (2, 0),
            dtype=torch.long,
        ),
        "shifts": torch.empty(
            (0, 3)
        ),
    }


def test_vector_representation_model() -> None:
    representation = DummyVectorRepresentation()

    head = TaskHead(
        2,
        n_out=1,
        hidden_layers=(),
    )

    model = RepresentationModel(
        representation,
        head=head,
    )

    x = torch.tensor(
        [
            [1.0, 2.0, 3.0],
            [2.0, 3.0, 4.0],
        ],
        requires_grad=True,
    )

    output = model(x)

    assert output.shape == (2, 1)
    assert model.in_features == 3
    assert model.out_features == 1

    assert all(
        not p.requires_grad
        for p in representation.parameters()
    )
    assert any(
        p.requires_grad
        for p in head.parameters()
    )

    output.sum().backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_pooled_graph_representation_model() -> None:
    representation = (
        DummyAtomRepresentation()
        .pool("mean")
    )

    assert representation.output_kind == "system"
    assert representation.out_features == 2

    model = RepresentationModel(
        representation,
        n_out=1,
        hidden_layers=(),
    )

    assert isinstance(
        model,
        BaseGNN,
    )

    output = model(
        make_graph()
    )

    assert output.shape == (2, 1)


def test_concat_graph_representation_model() -> None:
    representation = (
        DummyAtomRepresentation()
        .concat_atoms([0, 1])
    )

    assert representation.output_kind == "system"
    assert representation.out_features == 4

    model = RepresentationModel(
        representation,
        n_out=1,
        hidden_layers=(),
    )

    assert model.head.in_features == 4

    output = model(
        make_graph()
    )

    assert output.shape == (2, 1)


def test_representation_cache_has_no_task_dependency() -> None:
    representation = DummyVectorRepresentation()

    dataset = DictDataset(
        {
            "data": torch.randn(
                4,
                3,
            )
        }
    )

    cache = representation.cache(
        dataset,
        batch_size=2,
    )

    assert cache.features.shape == (
        4,
        2,
    )
    assert cache.jacobian is None
    assert cache.reference_indices is None


def test_model_cache_delegates_to_representation() -> None:
    representation = DummyVectorRepresentation()

    model = RepresentationModel(
        representation,
        n_out=1,
        hidden_layers=(),
    )

    dataset = DictDataset(
        {
            "data": torch.randn(
                4,
                3,
            )
        }
    )

    cache = model.cache(
        dataset,
        batch_size=2,
    )

    expected = representation.cache(
        dataset,
        batch_size=2,
    )

    torch.testing.assert_close(
        cache.features,
        expected.features,
    )


def test_representation_selective_jacobian_cache() -> None:
    representation = DummyVectorRepresentation()

    dataset = DictDataset(
        {
            "data": torch.randn(
                4,
                3,
            )
        }
    )

    cache = representation.cache(
        dataset,
        jacobian=True,
        descriptor_derivatives=(
            IdentityDescriptorDerivatives()
        ),
        jacobian_indices=torch.tensor(
            [1, 3]
        ),
        batch_size=1,
    )

    assert cache.jacobian is not None
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


def test_representation_cache_requires_frozen_representation() -> None:
    representation = DummyVectorRepresentation(
        freeze=False
    )

    dataset = DictDataset(
        {
            "data": torch.randn(
                4,
                3,
            )
        }
    )

    with pytest.raises(
        RuntimeError,
        match="frozen representation",
    ):
        representation.cache(
            dataset,
        )


def test_cached_representation_derivatives_require_ref_idx() -> None:
    derivatives = CachedRepresentationDerivatives(
        torch.zeros(
            2,
            3,
            2,
        )
    )

    with pytest.raises(
        ValueError,
        match="ref_idx",
    ):
        derivatives(
            torch.zeros(
                2,
                2,
            )
        )