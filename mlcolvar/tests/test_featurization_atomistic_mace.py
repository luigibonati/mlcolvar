from __future__ import annotations

from typing import Dict

import pytest
import torch
from torch import nn

from mlcolvar.featurization.atomistic import (
    AtomisticFeaturizer,
    MACEBackbone,
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
        del training, compute_force

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
    # layer_size = (l_max + 1)**2 * num_features = 8.
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


def make_backbone() -> MACEBackbone:
    """Create a MACE backbone with an explicit descriptor layout."""

    return MACEBackbone(
        model=DummyMACE(),
        num_layers=2,
        num_features=2,
        l_max=1,
    )


def test_mace_backbone_metadata_and_features(
    mace_test_data: Dict[str, torch.Tensor],
) -> None:
    """Extract invariant features and expose shared metadata."""

    backbone = make_backbone()
    output = backbone(mace_test_data)

    expected = torch.tensor(
        [
            [1.0, 2.0, 5.0, 6.0],
            [3.0, 4.0, 7.0, 8.0],
            [2.0, 4.0, 6.0, 8.0],
            [4.0, 6.0, 8.0, 10.0],
        ],
        dtype=torch.float64,
    )

    assert output.shape == (4, 4)
    assert output.dtype == torch.float64

    torch.testing.assert_close(
        output,
        expected,
    )

    assert backbone.out_features == 4
    assert backbone.sample_kind == "atom"
    assert backbone.full_neighbor_list

    assert backbone.num_layers == 2
    assert backbone.num_features == 2
    assert backbone.l_max == 1
    assert backbone.layer_size == 8
    assert backbone.required_input_features == 10

    assert backbone.atomic_numbers.tolist() == [1, 8]
    assert backbone.cutoff.item() == pytest.approx(5.0)
    assert backbone.buffer.item() == pytest.approx(0.0)
    assert backbone.long_range_cutoff.item() == pytest.approx(-1.0)


def test_mace_featurizer_mean_pooling_freeze_and_gradients(
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

    output = featurizer(data)

    expected = torch.tensor(
        [
            [2.0, 3.0, 6.0, 7.0],
            [3.0, 5.0, 7.0, 9.0],
        ],
        dtype=torch.float64,
    )

    assert output.shape == (2, 4)

    torch.testing.assert_close(
        output,
        expected,
    )

    output.sum().backward()

    gradient = node_features.grad

    assert gradient is not None
    assert gradient.shape == node_features.shape
    assert torch.isfinite(gradient).all()
    assert torch.count_nonzero(gradient) > 0

    # Freezing the pretrained model must disable parameter updates.
    assert all(
        not parameter.requires_grad
        for parameter in backbone.parameters()
    )

    # It must not disable gradients with respect to model inputs.
    assert backbone.model.weight.grad is None

    # Calling train() on the wrapper must keep a frozen backbone in eval mode.
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

    output = featurizer(mace_test_data)

    expected = torch.tensor(
        [
            [4.0, 6.0, 12.0, 14.0],
            [6.0, 10.0, 14.0, 18.0],
        ],
        dtype=torch.float64,
    )

    assert output.shape == (2, 4)

    torch.testing.assert_close(
        output,
        expected,
    )


def test_mace_featurizer_rejects_invalid_pooling() -> None:
    """Reject unsupported pooling operations."""

    with pytest.raises(
        ValueError,
        match="`pooling` must be either 'mean' or 'sum'",
    ):
        AtomisticFeaturizer(
            backbone=make_backbone(),
            pooling="max",
        )


def test_mace_featurizer_trace(
    mace_test_data: Dict[str, torch.Tensor],
) -> None:
    """Trace the frozen MACE backbone and shared pooling together."""

    featurizer = AtomisticFeaturizer(
        backbone=make_backbone(),
        pooling="sum",
        freeze=True,
    )

    featurizer.eval()

    expected = featurizer(mace_test_data)

    traced = torch.jit.trace(
        featurizer,
        example_inputs=(mace_test_data,),
        strict=False,
        check_trace=True,
    )

    output = traced(mace_test_data)

    assert output.shape == expected.shape
    assert output.dtype == expected.dtype

    torch.testing.assert_close(
        output,
        expected,
    )
