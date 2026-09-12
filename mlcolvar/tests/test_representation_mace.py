from __future__ import annotations

from typing import Dict

import pytest
import torch
from torch import nn

from mlcolvar.representation import (
    MACERepresentation,
    PoolReducer,
    RepresentationModel,
)


class DummyMACE(nn.Module):
    """Minimal MACE-like model used for unit testing."""

    def __init__(self) -> None:
        super().__init__()

        self.register_buffer(
            "atomic_numbers",
            torch.tensor(
                [1, 8],
                dtype=torch.int64,
            ),
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
            torch.tensor(
                2,
                dtype=torch.int64,
            ),
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
        del training
        del compute_force

        return {
            "node_feats": (
                data["node_feats"]
                * self.weight
            )
        }


@pytest.fixture
def mace_test_data() -> Dict[str, torch.Tensor]:
    """Create two two-atom graphs with two MACE interaction layers."""

    node_features = torch.zeros(
        (4, 10),
        dtype=torch.float64,
    )

    # Scalar features from interaction layer 1.
    node_features[:, 0:2] = torch.tensor(
        [
            [1.0, 2.0],
            [3.0, 4.0],
            [2.0, 4.0],
            [4.0, 6.0],
        ],
        dtype=torch.float64,
    )

    # Scalar features from interaction layer 2.
    #
    # For num_features=2 and l_max=1:
    #
    # layer_size =
    #     (l_max + 1)**2 * num_features
    #     = 8
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
    """Create a frozen MACE representation with explicit layout."""

    return MACERepresentation(
        model=DummyMACE(),
        num_layers=2,
        num_features=2,
        l_max=1,
        freeze=True,
    )


def test_mace_representation_metadata_and_features(
    mace_test_data: Dict[str, torch.Tensor],
) -> None:
    """Extract invariant features and expose unified metadata."""

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

    assert output.shape == (
        4,
        4,
    )

    assert (
        output.dtype
        == torch.float64
    )

    torch.testing.assert_close(
        output,
        expected,
    )

    # Unified representation metadata.
    assert (
        representation.input_kind
        == "graph"
    )

    assert (
        representation.output_kind
        == "atom"
    )

    assert (
        representation.in_features
        is None
    )

    assert (
        representation.out_features
        == 4
    )

    assert representation.freeze

    # MACE-specific descriptor layout.
    assert (
        representation.num_layers
        == 2
    )

    assert (
        representation.num_features
        == 2
    )

    assert (
        representation.l_max
        == 1
    )

    assert (
        representation.layer_size
        == 8
    )

    assert (
        representation.required_input_features
        == 10
    )

    # Deployment / graph metadata.
    assert (
        representation.atomic_numbers
        .tolist()
        == [1, 8]
    )

    assert (
        representation.cutoff.item()
        == pytest.approx(5.0)
    )

    assert (
        representation.buffer.item()
        == pytest.approx(0.0)
    )

    assert (
        representation
        .long_range_cutoff
        .item()
        == pytest.approx(-1.0)
    )

    assert (
        representation.full_neighbor_list
    )

    # Pretrained model is frozen.
    assert all(
        not parameter.requires_grad
        for parameter
        in representation.parameters()
    )


def test_mace_representation_preserves_input_gradients(
    mace_test_data: Dict[str, torch.Tensor],
) -> None:
    """Freezing MACE parameters must not disable input gradients."""

    node_features = (
        mace_test_data["node_feats"]
        .clone()
        .detach()
        .requires_grad_(True)
    )

    data = {
        **mace_test_data,
        "node_feats": node_features,
    }

    representation = make_representation()

    output = representation(
        data
    )

    assert output.shape == (
        4,
        4,
    )

    output.sum().backward()

    gradient = node_features.grad

    assert gradient is not None

    assert (
        gradient.shape
        == node_features.shape
    )

    assert torch.isfinite(
        gradient
    ).all()

    assert (
        torch.count_nonzero(
            gradient
        )
        > 0
    )

    assert all(
        not parameter.requires_grad
        for parameter
        in representation.parameters()
    )

    assert (
        representation
        .model
        .weight
        .grad
        is None
    )

    representation.train()

    # Frozen representations stay in inference mode.
    assert not (
        representation.training
    )

    assert not (
        representation.model.training
    )


def test_mace_mean_pooling(
    mace_test_data: Dict[str, torch.Tensor],
) -> None:
    """Pool atom-level MACE representations using the generic reducer."""

    representation = make_representation()

    reducer = PoolReducer(
        in_features=(
            representation.out_features
        ),
        pooling="mean",
    )

    atom_features = representation(
        mace_test_data
    )

    output = reducer(
        atom_features,
        mace_test_data,
    )

    expected = torch.tensor(
        [
            [2.0, 3.0, 6.0, 7.0],
            [3.0, 5.0, 7.0, 9.0],
        ],
        dtype=torch.float64,
    )

    assert output.shape == (
        2,
        4,
    )

    torch.testing.assert_close(
        output,
        expected,
    )


def test_mace_sum_pooling(
    mace_test_data: Dict[str, torch.Tensor],
) -> None:
    """Support sum pooling through the generic reducer."""

    representation = make_representation()

    reducer = PoolReducer(
        in_features=(
            representation.out_features
        ),
        pooling="sum",
    )

    output = reducer(
        representation(
            mace_test_data
        ),
        mace_test_data,
    )

    expected = torch.tensor(
        [
            [4.0, 6.0, 12.0, 14.0],
            [6.0, 10.0, 14.0, 18.0],
        ],
        dtype=torch.float64,
    )

    assert output.shape == (
        2,
        4,
    )

    torch.testing.assert_close(
        output,
        expected,
    )


def test_pool_reducer_rejects_invalid_pooling() -> None:
    """Reject unsupported generic pooling operations."""

    with pytest.raises(
        ValueError,
        match="`pooling` must be 'mean' or 'sum'",
    ):
        PoolReducer(
            in_features=4,
            pooling="max",
        )


def test_mace_representation_model_default_pooling(
    mace_test_data: Dict[str, torch.Tensor],
) -> None:
    """Atom-level representations are pooled before the task head."""

    representation = make_representation()

    model = RepresentationModel(
        representation,
        n_out=2,
        hidden_layers=(),
    )

    output = model(
        mace_test_data
    )

    assert (
        model.representation
        is representation
    )

    assert isinstance(
        model.pre_head,
        PoolReducer,
    )

    assert output.shape == (
        2,
        2,
    )

    # Only the downstream task head should remain trainable.
    assert all(
        not parameter.requires_grad
        for parameter
        in representation.parameters()
    )

    assert any(
        parameter.requires_grad
        for parameter
        in model.head.parameters()
    )


def test_mace_representation_model_preserves_input_gradients(
    mace_test_data: Dict[str, torch.Tensor],
) -> None:
    """Preserve gradients through representation, pooling and task head."""

    node_features = (
        mace_test_data["node_feats"]
        .clone()
        .detach()
        .requires_grad_(True)
    )

    data = {
        **mace_test_data,
        "node_feats": node_features,
    }

    representation = make_representation()

    model = RepresentationModel(
        representation,
        n_out=1,
        hidden_layers=(),
    )

    output = model(
        data
    )

    assert output.shape == (
        2,
        1,
    )

    output.sum().backward()

    gradient = node_features.grad

    assert gradient is not None

    assert torch.isfinite(
        gradient
    ).all()

    assert (
        torch.count_nonzero(
            gradient
        )
        > 0
    )

    assert (
        representation
        .model
        .weight
        .grad
        is None
    )

    assert any(
        parameter.grad is not None
        for parameter
        in model.head.parameters()
    )


def test_mace_representation_model_sum_pooling(
    mace_test_data: Dict[str, torch.Tensor],
) -> None:
    """Allow RepresentationModel to explicitly select sum pooling."""

    representation = make_representation()

    model = RepresentationModel(
        representation,
        n_out=2,
        hidden_layers=(),
        mode="pooled",
        pooling="sum",
    )

    assert isinstance(
        model.pre_head,
        PoolReducer,
    )

    assert (
        model.pre_head.pooling
        == "sum"
    )

    output = model(
        mace_test_data
    )

    assert output.shape == (
        2,
        2,
    )


def test_mace_representation_model_trace(
    mace_test_data: Dict[str, torch.Tensor],
) -> None:
    """Trace representation + reducer + task head."""

    model = RepresentationModel(
        make_representation(),
        n_out=2,
        hidden_layers=(),
        mode="pooled",
        pooling="sum",
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

    assert (
        output.shape
        == expected.shape
    )

    assert (
        output.dtype
        == expected.dtype
    )

    torch.testing.assert_close(
        output,
        expected,
    )