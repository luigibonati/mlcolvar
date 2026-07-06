from typing import Dict, Tuple

import pytest
import torch
from torch import nn

from mlcolvar.integrations.atomistic import AtomisticFeaturizer
from mlcolvar.integrations.backbones.mace import MACEBackbone


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
    """Create two graphs containing two atoms each.

    Descriptor layout:

    - num_layers = 2
    - num_features = 2
    - l_max = 1
    - layer_size = (1 + 1)^2 * 2 = 8
    - required width = 8 + 2 = 10
    """

    node_features = torch.zeros(
        (4, 10),
        dtype=torch.float64,
    )

    # Scalar invariant block from the first interaction layer.
    node_features[:, 0:2] = torch.tensor(
        [
            [1.0, 2.0],
            [3.0, 4.0],
            [2.0, 4.0],
            [4.0, 6.0],
        ],
        dtype=torch.float64,
    )

    # Scalar invariant block from the second interaction layer.
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
            dtype=torch.int64,
        ),
        "ptr": torch.tensor(
            [0, 2, 4],
            dtype=torch.int64,
        ),
    }


def create_backbone() -> MACEBackbone:
    """Create a MACE backbone with an explicit descriptor layout."""

    return MACEBackbone(
        model=DummyMACE(),
        num_layers=2,
        num_features=2,
        l_max=1,
    )


def create_featurizer(
    pooling: str = "mean",
    freeze: bool = True,
) -> Tuple[MACEBackbone, AtomisticFeaturizer]:
    """Create a MACE backbone and shared atomistic featurizer."""

    backbone = create_backbone()

    featurizer = AtomisticFeaturizer(
        backbone=backbone,
        pooling=pooling,
        freeze=freeze,
    )

    return backbone, featurizer


def test_mace_backbone(
    mace_test_data: Dict[str, torch.Tensor],
) -> None:
    """Test extraction of atom-level invariant MACE features."""

    backbone = create_backbone()

    output = backbone(
        mace_test_data
    )

    reference = torch.tensor(
        [
            [1.0, 2.0, 5.0, 6.0],
            [3.0, 4.0, 7.0, 8.0],
            [2.0, 4.0, 6.0, 8.0],
            [4.0, 6.0, 8.0, 10.0],
        ],
        dtype=torch.float64,
    )

    assert output.shape == (4, 4)
    assert torch.allclose(
        output,
        reference,
    )

    assert backbone.out_features == 4
    assert backbone.sample_kind == "atom"
    assert backbone.layer_size == 8
    assert backbone.required_input_features == 10

    assert torch.equal(
        backbone.atomic_numbers,
        torch.tensor(
            [1, 8],
            dtype=torch.int64,
        ),
    )

    assert backbone.cutoff.item() == pytest.approx(5.0)


@pytest.mark.parametrize(
    ("pooling", "reference"),
    [
        (
            "mean",
            torch.tensor(
                [
                    [2.0, 3.0, 6.0, 7.0],
                    [3.0, 5.0, 7.0, 9.0],
                ],
                dtype=torch.float64,
            ),
        ),
        (
            "sum",
            torch.tensor(
                [
                    [4.0, 6.0, 12.0, 14.0],
                    [6.0, 10.0, 14.0, 18.0],
                ],
                dtype=torch.float64,
            ),
        ),
    ],
)
def test_atomistic_featurizer_pooling(
    mace_test_data: Dict[str, torch.Tensor],
    pooling: str,
    reference: torch.Tensor,
) -> None:
    """Test shared atom-to-graph pooling for MACE features."""

    backbone, featurizer = create_featurizer(
        pooling=pooling,
    )

    output = featurizer(
        mace_test_data
    )

    assert output.shape == (2, 4)

    assert torch.allclose(
        output,
        reference,
    )

    assert featurizer.out_features == 4
    assert featurizer.sample_kind == "atom"
    assert featurizer.pooling == pooling

    assert torch.equal(
        featurizer.atomic_numbers,
        backbone.atomic_numbers,
    )

    assert torch.equal(
        featurizer.feature_dim,
        backbone.feature_dim,
    )


def test_atomistic_featurizer_freeze() -> None:
    """Test that a frozen MACE backbone stays in evaluation mode."""

    backbone, featurizer = create_featurizer(
        freeze=True,
    )

    assert all(
        not parameter.requires_grad
        for parameter in backbone.parameters()
    )

    assert all(
        not parameter.requires_grad
        for parameter in backbone.model.parameters()
    )

    featurizer.train()

    assert featurizer.training

    # A frozen backbone must stay in evaluation mode.
    assert not backbone.training
    assert not backbone.model.training


def test_atomistic_featurizer_unfrozen() -> None:
    """Test that an unfrozen MACE backbone follows training mode."""

    backbone, featurizer = create_featurizer(
        freeze=False,
    )

    assert all(
        parameter.requires_grad
        for parameter in backbone.parameters()
    )

    featurizer.train()

    assert featurizer.training
    assert backbone.training
    assert backbone.model.training

    featurizer.eval()

    assert not featurizer.training
    assert not backbone.training
    assert not backbone.model.training


def test_atomistic_featurizer_preserves_input_gradients(
    mace_test_data: Dict[str, torch.Tensor],
) -> None:
    """Test that freezing MACE preserves gradients with respect to inputs."""

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

    _, featurizer = create_featurizer(
        freeze=True,
    )

    output = featurizer(data)

    output.sum().backward()

    assert node_features.grad is not None

    assert (
        torch.count_nonzero(
            node_features.grad
        )
        > 0
    )


def test_mace_backbone_trace(
    mace_test_data: Dict[str, torch.Tensor],
) -> None:
    """Test TorchScript tracing of the model-specific backbone."""

    backbone = create_backbone()
    backbone.eval()

    traced = torch.jit.trace(
        backbone,
        example_inputs=(
            mace_test_data,
        ),
        strict=False,
    )

    reference = backbone(
        mace_test_data
    )

    output = traced(
        mace_test_data
    )

    assert torch.allclose(
        output,
        reference,
    )


def test_atomistic_featurizer_trace(
    mace_test_data: Dict[str, torch.Tensor],
) -> None:
    """Test TorchScript tracing of backbone and graph pooling together."""

    _, featurizer = create_featurizer(
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

    reference = featurizer(
        mace_test_data
    )

    output = traced(
        mace_test_data
    )

    assert torch.allclose(
        output,
        reference,
    )