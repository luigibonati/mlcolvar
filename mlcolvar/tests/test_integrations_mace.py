from __future__ import annotations

from typing import Dict

import pytest
import torch
from torch import nn

from mlcolvar.integrations import AtomisticFeaturizer, MACEBackbone


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
        _ = training
        _ = compute_force

        return {
            "node_feats": (
                data["node_feats"]
                * self.weight
            ),
        }


@pytest.fixture
def mace_test_data() -> Dict[str, torch.Tensor]:
    """Create two two-atom graphs with two MACE interaction layers."""

    node_features = torch.zeros(
        (
            4,
            10,
        ),
        dtype=torch.float64,
    )

    node_features[:, 0:2] = torch.tensor(
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
            [
                0,
                0,
                1,
                1,
            ],
            dtype=torch.long,
        ),
        "ptr": torch.tensor(
            [
                0,
                2,
                4,
            ],
            dtype=torch.long,
        ),
    }


def make_backbone() -> MACEBackbone:
    """Create a MACE backbone with an explicit descriptor layout."""

    return MACEBackbone(
        model=DummyMACE(),
        num_layers=2,
        num_features=2,
        l_max=1,
    )


def test_mace_backbone(
    mace_test_data: Dict[str, torch.Tensor],
) -> None:
    """Extract invariant features and expose shared metadata."""

    backbone = make_backbone()

    output = backbone(
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

    assert torch.allclose(
        output,
        expected,
    )

    assert backbone.out_features == 4
    assert backbone.sample_kind == "atom"
    assert backbone.layer_size == 8
    assert backbone.required_input_features == 10
    assert backbone.atomic_numbers.tolist() == [
        1,
        8,
    ]

    assert backbone.cutoff.item() == pytest.approx(
        5.0
    )


def test_mace_featurizer_pooling_freeze_and_gradients(
    mace_test_data: Dict[str, torch.Tensor],
) -> None:
    """Pool MACE features while preserving input gradients."""

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

    backbone = make_backbone()

    featurizer = AtomisticFeaturizer(
        backbone=backbone,
        pooling="mean",
        freeze=True,
    )

    output = featurizer(
        data
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

    assert torch.allclose(
        output,
        expected,
    )

    output.sum().backward()

    assert node_features.grad is not None

    assert torch.count_nonzero(
        node_features.grad
    ) > 0

    assert all(
        not parameter.requires_grad
        for parameter in backbone.parameters()
    )

    featurizer.train()

    assert featurizer.training
    assert not backbone.training
    assert not backbone.model.training


def test_mace_featurizer_sum_pooling(
    mace_test_data: Dict[str, torch.Tensor],
) -> None:
    """Support sum pooling through the shared atomistic featurizer."""

    featurizer = AtomisticFeaturizer(
        backbone=make_backbone(),
        pooling="sum",
        freeze=True,
    )

    output = featurizer(
        mace_test_data
    )

    expected = torch.tensor(
        [
            [4.0, 6.0, 12.0, 14.0],
            [6.0, 10.0, 14.0, 18.0],
        ],
        dtype=torch.float64,
    )

    assert torch.allclose(
        output,
        expected,
    )


def test_mace_featurizer_trace(
    mace_test_data: Dict[str, torch.Tensor],
) -> None:
    """Trace the backbone and shared pooling together."""

    featurizer = AtomisticFeaturizer(
        backbone=make_backbone(),
        pooling="sum",
        freeze=True,
    )

    featurizer.eval()

    traced = torch.jit.trace(
        featurizer,
        example_inputs=(
            mace_test_data,
        ),
        strict=False,
    )

    expected = featurizer(
        mace_test_data
    )

    output = traced(
        mace_test_data
    )

    assert torch.allclose(
        output,
        expected,
    )
