from typing import Dict

import pytest
import torch
from torch import nn

from mlcolvar.integrations.mace_featurizer import MACEFeaturizer


class DummyMACE(nn.Module):
    """Minimal MACE-like model used for unit testing."""

    def __init__(self) -> None:
        super().__init__()

        self.register_buffer(
            "atomic_numbers",
            torch.tensor([1, 8], dtype=torch.int64),
        )
        self.register_buffer(
            "r_max",
            torch.tensor(5.0, dtype=torch.float64),
        )
        self.register_buffer(
            "num_interactions",
            torch.tensor(2, dtype=torch.int64),
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
            "node_feats": data["node_feats"] * self.weight,
        }


@pytest.fixture
def mace_test_data() -> Dict[str, torch.Tensor]:
    """Create two graphs containing two atoms each."""

    # num_layers=2, num_features=2, l_max=1:
    #
    # first layer width: 8
    # final invariant block width: 2
    # total width: 10
    node_features = torch.zeros(
        (4, 10),
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
            [0, 0, 1, 1],
            dtype=torch.int64,
        ),
        "ptr": torch.tensor(
            [0, 2, 4],
            dtype=torch.int64,
        ),
    }


def create_featurizer(
    pooling: str = "mean",
    freeze: bool = True,
) -> MACEFeaturizer:
    """Create a MACE featurizer with an explicit descriptor layout."""

    return MACEFeaturizer(
        model=DummyMACE(),
        num_layers=2,
        num_features=2,
        l_max=1,
        pooling=pooling,
        freeze=freeze,
    )


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
def test_mace_featurizer(
    mace_test_data: Dict[str, torch.Tensor],
    pooling: str,
    reference: torch.Tensor,
) -> None:
    """Test invariant feature extraction and graph pooling."""

    featurizer = create_featurizer(
        pooling=pooling,
    )

    output = featurizer(
        mace_test_data
    )

    assert output.shape == (2, 4)
    assert torch.allclose(output, reference)

    assert featurizer.out_features == 4
    assert featurizer.layer_size == 8
    assert featurizer.required_input_features == 10


def test_mace_featurizer_freeze() -> None:
    """Test that a frozen MACE backbone stays in evaluation mode."""

    featurizer = create_featurizer(
        freeze=True,
    )

    assert all(
        not parameter.requires_grad
        for parameter in featurizer.model.parameters()
    )

    featurizer.train()

    assert featurizer.training
    assert not featurizer.model.training


def test_mace_featurizer_preserves_input_gradients(
    mace_test_data: Dict[str, torch.Tensor],
) -> None:
    """Test that freezing MACE preserves input gradients."""

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

    featurizer = create_featurizer()

    featurizer(data).sum().backward()

    assert node_features.grad is not None
    assert torch.count_nonzero(node_features.grad) > 0


def test_mace_featurizer_trace(
    mace_test_data: Dict[str, torch.Tensor],
) -> None:
    """Test TorchScript tracing."""

    featurizer = create_featurizer(
        pooling="sum",
    )
    featurizer.eval()

    traced = torch.jit.trace(
        featurizer,
        example_inputs=(mace_test_data,),
        strict=False,
    )

    assert torch.allclose(
        traced(mace_test_data),
        featurizer(mace_test_data),
    )