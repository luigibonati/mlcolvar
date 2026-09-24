from typing import Dict

import torch
from torch import nn

from mlcolvar.representation import (
    MACERepresentation,
    RepresentationModel,
)
import pytest


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
            torch.tensor(5.0, dtype=torch.float64),
        )
        self.register_buffer(
            "num_interactions",
            torch.tensor(2),
        )

        self.weight = nn.Parameter(
            torch.tensor(1.0, dtype=torch.float64)
        )

    def forward(
        self,
        data: Dict[str, torch.Tensor],
        training: bool = False,
        compute_force: bool = False,
    ) -> Dict[str, torch.Tensor]:
        return {
            "node_feats": (
                data["node_feats"] * self.weight
            )
        }


def make_data() -> Dict[str, torch.Tensor]:
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


def test_mace_representation():
    representation = make_representation()

    output = representation(
        make_data()
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

    assert representation.output_kind == "atom"
    assert representation.out_features == 4
    assert representation.atomic_numbers.tolist() == [1, 8]
    assert representation.cutoff.item() == pytest.approx(5.0)


def test_mace_preserves_input_gradients():
    representation = make_representation()

    data = make_data()
    data["node_feats"].requires_grad_(True)

    representation(
        data
    ).sum().backward()

    assert data["node_feats"].grad is not None
    assert torch.isfinite(
        data["node_feats"].grad
    ).all()

    assert all(
        parameter.grad is None
        for parameter
        in representation.model.parameters()
    )

    assert all(
        not parameter.requires_grad
        for parameter
        in representation.model.parameters()
    )


def test_mace_representation_model():
    representation = (
        make_representation()
        .pool("mean")
    )

    model = RepresentationModel(
        representation,
        n_out=1,
        hidden_layers=(),
    )

    output = model(
        make_data()
    )

    assert representation.output_kind == "system"
    assert output.shape == (2, 1)

    assert any(
        parameter.requires_grad
        for parameter
        in model.head.parameters()
    )


def test_mace_representation_trace():
    model = RepresentationModel(
        make_representation().pool("mean"),
        n_out=1,
        hidden_layers=(),
    ).eval()

    data = make_data()

    expected = model(data)

    traced = torch.jit.trace(
        model,
        example_inputs=(data,),
        strict=False,
    )

    output = traced(data)

    torch.testing.assert_close(
        output,
        expected,
    )
    
def test_mace_representation_invalid_node_features():
    representation = make_representation()

    data = make_data()
    data["node_feats"] = torch.zeros(
        4,
        9,
        dtype=torch.float64,
    )

    with pytest.raises(
        RuntimeError,
        match="incompatible with the configured descriptor layout",
    ):
        representation(data)