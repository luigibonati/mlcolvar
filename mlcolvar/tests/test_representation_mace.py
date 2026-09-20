from typing import Dict

import pytest
import torch
from torch import nn

from mlcolvar.representation import (
    MACERepresentation,
    RepresentationModel,
)


class DummyMACE(nn.Module):
    """Minimal MACE-like model for testing."""

    def __init__(self) -> None:
        super().__init__()

        self.register_buffer(
            "atomic_numbers",
            torch.tensor([1, 8]),
        )
        self.register_buffer(
            "r_max",
            torch.tensor(
                5.0,
                dtype=torch.float64,
            ),
        )
        self.register_buffer(
            "num_interactions",
            torch.tensor(2),
        )

        self.weight = nn.Parameter(
            torch.tensor(
                1.0,
                dtype=torch.float64,
            )
        )

    def forward(
        self,
        data: Dict[str, torch.Tensor],
        training: bool = False,
        compute_force: bool = False,
    ) -> Dict[str, torch.Tensor]:
        del training, compute_force

        return {
            "node_feats": (
                data["node_feats"]
                * self.weight
            )
        }


@pytest.fixture
def mace_test_data() -> Dict[str, torch.Tensor]:
    node_features = torch.zeros(
        4,
        10,
        dtype=torch.float64,
    )

    node_features[:, :2] = torch.tensor(
        [
            [1.0, 2.0],
            [3.0, 4.0],
            [2.0, 4.0],
            [4.0, 6.0],
        ],
        dtype=torch.float64,
    )

    node_features[:, 8:10] = torch.tensor(
        [
            [5.0, 6.0],
            [7.0, 8.0],
            [6.0, 8.0],
            [8.0, 10.0],
        ],
        dtype=torch.float64,
    )

    return {
        "node_feats": node_features,
        "batch": torch.tensor(
            [0, 0, 1, 1],
            dtype=torch.long,
        ),
        "ptr": torch.tensor(
            [0, 2, 4],
            dtype=torch.long,
        ),
    }


def make_representation() -> MACERepresentation:
    return MACERepresentation(
        model=DummyMACE(),
        num_layers=2,
        num_features=2,
        l_max=1,
        freeze=True,
    )


def test_mace_representation(
    mace_test_data,
) -> None:
    representation = make_representation()
    output = representation(
        mace_test_data
    )

    expected = torch.tensor(
        [
            [1.0, 2.0, 5.0, 6.0],
            [3.0, 4.0, 7.0, 8.0],
            [2.0, 4.0, 6.0, 8.0],
            [4.0, 6.0, 8.0, 10.0],
        ],
        dtype=torch.float64,
    )

    torch.testing.assert_close(
        output,
        expected,
    )

    assert representation.input_kind == "graph"
    assert representation.output_kind == "atom"
    assert representation.out_features == 4
    assert representation.num_layers == 2
    assert representation.num_features == 2
    assert representation.l_max == 1
    assert representation.atomic_numbers.tolist() == [1, 8]
    assert (
        representation.cutoff.item()
        == pytest.approx(5.0)
    )
    assert representation.freeze

    assert all(
        not p.requires_grad
        for p in representation.parameters()
    )


def test_mace_preserves_input_gradients(
    mace_test_data,
) -> None:
    features = (
        mace_test_data["node_feats"]
        .clone()
        .requires_grad_(True)
    )

    data = {
        **mace_test_data,
        "node_feats": features,
    }

    representation = make_representation()

    representation(
        data
    ).sum().backward()

    assert features.grad is not None
    assert torch.isfinite(
        features.grad
    ).all()
    assert torch.count_nonzero(
        features.grad
    ) > 0

    assert (
        representation.model.weight.grad
        is None
    )

    representation.train()

    assert not representation.training
    assert not representation.model.training


@pytest.mark.parametrize(
    ("pooling", "expected"),
    [
        (
            "mean",
            [
                [2.0, 3.0, 6.0, 7.0],
                [3.0, 5.0, 7.0, 9.0],
            ],
        ),
        (
            "sum",
            [
                [4.0, 6.0, 12.0, 14.0],
                [6.0, 10.0, 14.0, 18.0],
            ],
        ),
    ],
)
def test_mace_pooling(
    mace_test_data,
    pooling,
    expected,
) -> None:
    representation = (
        make_representation()
        .pool(pooling)
    )

    output = representation(
        mace_test_data
    )

    torch.testing.assert_close(
        output,
        torch.tensor(
            expected,
            dtype=torch.float64,
        ),
    )

    assert (
        representation.output_kind
        == "system"
    )
    assert output.shape == (2, 4)


def test_invalid_pooling() -> None:
    with pytest.raises(
        ValueError,
        match="pooling",
    ):
        make_representation().pool(
            "max"
        )


@pytest.mark.parametrize(
    "pooling",
    ["mean", "sum"],
)
def test_mace_representation_model(
    mace_test_data,
    pooling,
) -> None:
    representation = (
        make_representation()
        .pool(pooling)
    )

    model = RepresentationModel(
        representation,
        n_out=2,
        hidden_layers=(),
    )

    output = model(
        mace_test_data
    )

    assert output.shape == (2, 2)

    assert all(
        not p.requires_grad
        for p in representation.parameters()
    )

    assert any(
        p.requires_grad
        for p in model.head.parameters()
    )


def test_mace_model_preserves_input_gradients(
    mace_test_data,
) -> None:
    features = (
        mace_test_data["node_feats"]
        .clone()
        .requires_grad_(True)
    )

    data = {
        **mace_test_data,
        "node_feats": features,
    }

    atom_representation = (
        make_representation()
    )

    model = RepresentationModel(
        atom_representation.pool(
            "mean"
        ),
        n_out=1,
        hidden_layers=(),
    )

    model(
        data
    ).sum().backward()

    assert features.grad is not None
    assert torch.isfinite(
        features.grad
    ).all()

    assert (
        atom_representation
        .model
        .weight
        .grad
        is None
    )

    assert any(
        p.grad is not None
        for p in model.head.parameters()
    )


def test_mace_representation_model_trace(
    mace_test_data,
) -> None:
    model = RepresentationModel(
        make_representation().pool(
            "sum"
        ),
        n_out=2,
        hidden_layers=(),
    ).eval()

    expected = model(
        mace_test_data
    )

    traced = torch.jit.trace(
        model,
        example_inputs=(
            mace_test_data,
        ),
        strict=False,
        check_trace=True,
    )

    output = traced(
        mace_test_data
    )

    torch.testing.assert_close(
        output,
        expected,
    )