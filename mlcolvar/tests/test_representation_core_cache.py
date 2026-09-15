from __future__ import annotations

from typing import Dict, Optional

import pytest
import torch
from torch import nn

from mlcolvar.core import BaseGNN
from mlcolvar.core.loss.utils.smart_derivatives import SmartDerivatives
from mlcolvar.data import DictDataset
from mlcolvar.representation import (
    CachedRepresentationDerivatives,
    ConcatReducer,
    GraphRepresentation,
    PoolReducer,
    RepresentationModel,
    TaskHead,
    TensorRepresentation,
    precompute_representation_cache,
)


class DummyTensorRepresentation(TensorRepresentation):
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


def test_tensor_representation_model_direct() -> None:
    representation = DummyTensorRepresentation()

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


def test_atom_representation_default_is_pooled() -> None:
    representation = DummyAtomRepresentation()

    model = RepresentationModel(
        representation,
        n_out=1,
        hidden_layers=(),
    )

    assert isinstance(
        model,
        BaseGNN,
    )
    assert isinstance(
        model.pre_head,
        PoolReducer,
    )

    output = model(
        make_graph()
    )

    assert output.shape == (2, 1)


def test_atom_representation_concat() -> None:
    representation = DummyAtomRepresentation()

    model = RepresentationModel(
        representation,
        n_out=1,
        hidden_layers=(),
        mode="concat",
        selected_atom_indices=[
            0,
            1,
        ],
    )

    assert isinstance(
        model.pre_head,
        ConcatReducer,
    )
    assert model.head.in_features == 4

    output = model(
        make_graph()
    )

    assert output.shape == (2, 1)


def test_system_representation_rejects_atom_modes() -> None:
    representation = DummyTensorRepresentation()

    with pytest.raises(
        ValueError,
        match="atom-level",
    ):
        RepresentationModel(
            representation,
            mode="pooled",
        )


def test_generic_cache_has_no_task_dependency() -> None:
    representation = DummyTensorRepresentation()

    dataset = DictDataset(
        {
            "data": torch.randn(
                4,
                3,
            )
        }
    )

    cache = precompute_representation_cache(
        representation,
        dataset,
        compute_jacobian=False,
        batch_size=2,
    )

    assert cache.features.shape == (
        4,
        2,
    )
    assert cache.jacobian is None
    assert cache.reference_indices is None


def test_generic_selective_jacobian_cache() -> None:
    representation = DummyTensorRepresentation()

    dataset = DictDataset(
        {
            "data": torch.randn(
                4,
                3,
            )
        }
    )

    cache = precompute_representation_cache(
        representation,
        dataset,
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
