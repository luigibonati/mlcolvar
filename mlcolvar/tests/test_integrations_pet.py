"""Tests for the PET integration."""

from __future__ import annotations

from typing import Dict, List, Optional

import pytest
import torch
from torch import nn


# PET is an optional dependency. Skip this test module cleanly when the
# metatensor/metatomic stack is not installed.
metatensor_torch = pytest.importorskip("metatensor.torch")
metatomic_torch = pytest.importorskip("metatomic.torch")

Labels = metatensor_torch.Labels
TensorBlock = metatensor_torch.TensorBlock
TensorMap = metatensor_torch.TensorMap

ModelOutput = metatomic_torch.ModelOutput
NeighborListOptions = metatomic_torch.NeighborListOptions
System = metatomic_torch.System

from mlcolvar.integrations import (  # noqa: E402
    AtomisticFeaturizer,
    PETBackbone,
)


class DummyPET(nn.Module):
    """Minimal native PET-like model using the public metatomic API."""

    def __init__(
        self,
        cutoff: float = 2.5,
    ) -> None:
        super().__init__()

        self.atomic_types = [1, 8]
        self.cutoff = float(cutoff)

        self.d_node = 2
        self.d_pet = 2
        self.num_readout_layers = 2

        self.scale = nn.Parameter(
            torch.tensor(
                1.0,
                dtype=torch.float64,
            )
        )

        self._neighbor_options = NeighborListOptions(
            cutoff=self.cutoff,
            full_list=True,
            strict=True,
        )

    def supported_outputs(self):
        """Expose PET's atom-level feature output."""
        return {
            "feature": ModelOutput(
                sample_kind="atom",
            )
        }

    def requested_neighbor_lists(self):
        """Request one strict full neighbor list."""
        return [
            self._neighbor_options
        ]

    def forward(
        self,
        systems: List[System],
        outputs: Dict[str, ModelOutput],
        selected_atoms: Optional[Labels] = None,
    ):
        """Return deterministic atom-level features."""
        assert "feature" in outputs
        assert selected_atoms is None

        values_list: List[torch.Tensor] = []
        sample_rows: List[List[int]] = []

        for system_index, system in enumerate(systems):
            # Verify that PETBackbone attached the requested neighbor list.
            _ = system.get_neighbor_list(
                self._neighbor_options
            )

            positions = system.positions
            atom_types = system.types.to(
                dtype=positions.dtype,
            ).reshape(-1, 1)

            # Four base features:
            # [x, y, z, atomic_number]
            base_features = torch.cat(
                [
                    positions,
                    atom_types,
                ],
                dim=1,
            )

            # d_node=2, d_pet=2, num_readout_layers=2
            # gives 2 * (2 + 2) = 8 output features.
            features = torch.cat(
                [
                    base_features,
                    base_features,
                ],
                dim=1,
            )

            values_list.append(
                features * self.scale
            )

            for atom_index in range(
                positions.size(0)
            ):
                sample_rows.append(
                    [
                        system_index,
                        atom_index,
                    ]
                )

        values = torch.cat(
            values_list,
            dim=0,
        )

        samples = Labels(
            names=[
                "system",
                "atom",
            ],
            values=torch.tensor(
                sample_rows,
                dtype=torch.int32,
                device=values.device,
            ),
        )

        properties = Labels(
            names=["feature"],
            values=torch.arange(
                values.size(1),
                dtype=torch.int32,
                device=values.device,
            ).reshape(-1, 1),
        )

        block = TensorBlock(
            values=values,
            samples=samples,
            components=[],
            properties=properties,
        )

        return {
            "feature": TensorMap(
                keys=Labels.single(),
                blocks=[block],
            )
        }


class DummyPETWrapper(nn.Module):
    """Minimal LLPR-like wrapper exposing the native model as `.model`."""

    def __init__(
        self,
        model: nn.Module,
    ) -> None:
        super().__init__()
        self.model = model


def _create_pet_test_data() -> Dict[str, torch.Tensor]:
    """Create a batch containing two non-periodic two-atom systems."""

    positions = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
        ],
        dtype=torch.float64,
    )

    # PET atomic-number order: [H, O] = [1, 8].
    node_attrs = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=torch.float64,
    )

    # Full directed neighbor list:
    # system 0: 0 -> 1 and 1 -> 0
    # system 1: 2 -> 3 and 3 -> 2
    edge_index = torch.tensor(
        [
            [0, 1, 2, 3],
            [1, 0, 3, 2],
        ],
        dtype=torch.int64,
    )

    return {
        "positions": positions,
        "node_attrs": node_attrs,
        "edge_index": edge_index,
        "unit_shifts": torch.zeros(
            (4, 3),
            dtype=torch.float64,
        ),
        "batch": torch.tensor(
            [0, 0, 1, 1],
            dtype=torch.int64,
        ),
        "ptr": torch.tensor(
            [0, 2, 4],
            dtype=torch.int64,
        ),
        "cell": torch.zeros(
            (2, 3, 3),
            dtype=torch.float64,
        ),
        "pbc": torch.zeros(
            (2, 3),
            dtype=torch.bool,
        ),
    }


def test_pet_backbone_metadata() -> None:
    """PET metadata should be mapped to the shared backbone interface."""

    native_pet = DummyPET()

    backbone = PETBackbone(
        model=native_pet,
    )

    assert backbone.model is native_pet
    assert backbone.out_features == 8
    assert backbone.sample_kind == "atom"
    assert backbone.full_neighbor_list
    assert backbone.neighbor_full_list
    assert backbone.neighbor_strict

    assert backbone.atomic_numbers.tolist() == [
        1,
        8,
    ]

    assert backbone.cutoff.item() == pytest.approx(
        2.5
    )

    assert backbone.neighbor_cutoff == pytest.approx(
        2.5
    )


def test_pet_wrapper_is_unwrapped() -> None:
    """An LLPR-like wrapper should be unwrapped automatically."""

    native_pet = DummyPET()

    wrapped_pet = DummyPETWrapper(
        native_pet
    )

    backbone = PETBackbone(
        model=wrapped_pet,
    )

    assert backbone.model is native_pet


def test_invalid_pet_model() -> None:
    """Models without the public PET interface should be rejected."""

    with pytest.raises(
        ValueError,
        match="Could not find a native metatrain PET model",
    ):
        PETBackbone(
            model=nn.Linear(
                2,
                2,
            )
        )


def test_pet_neighbor_list_conversion() -> None:
    """mlcolvar edges should become metatomic distance vectors."""

    data = _create_pet_test_data()

    backbone = PETBackbone(
        model=DummyPET(),
    )

    neighbors = backbone._build_neighbor_list(
        data=data,
        positions=data["positions"][:2],
        cell=data["cell"][0],
        start=0,
        end=2,
    )

    expected_samples = torch.tensor(
        [
            [0, 1, 0, 0, 0],
            [1, 0, 0, 0, 0],
        ],
        dtype=torch.int32,
    )

    expected_vectors = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
        ],
        dtype=torch.float64,
    )

    assert torch.equal(
        neighbors.samples.values.cpu(),
        expected_samples,
    )

    assert torch.allclose(
        neighbors.values.squeeze(-1),
        expected_vectors,
    )


def test_pet_backbone_forward() -> None:
    """PETBackbone should return one feature row per atom."""

    data = _create_pet_test_data()

    backbone = PETBackbone(
        model=DummyPET(),
    )

    output = backbone(
        data
    )

    reference = torch.tensor(
        [
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, 8.0, 1.0, 0.0, 0.0, 8.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            [0.0, 2.0, 0.0, 8.0, 0.0, 2.0, 0.0, 8.0],
        ],
        dtype=torch.float64,
    )

    assert output.shape == (
        4,
        8,
    )

    assert torch.allclose(
        output,
        reference,
    )


@pytest.mark.parametrize(
    ("pooling", "reference"),
    [
        (
            "mean",
            torch.tensor(
                [
                    [0.5, 0.0, 0.0, 4.5, 0.5, 0.0, 0.0, 4.5],
                    [0.0, 1.0, 0.0, 4.5, 0.0, 1.0, 0.0, 4.5],
                ],
                dtype=torch.float64,
            ),
        ),
        (
            "sum",
            torch.tensor(
                [
                    [1.0, 0.0, 0.0, 9.0, 1.0, 0.0, 0.0, 9.0],
                    [0.0, 2.0, 0.0, 9.0, 0.0, 2.0, 0.0, 9.0],
                ],
                dtype=torch.float64,
            ),
        ),
    ],
)
def test_pet_featurizer_pooling(
    pooling: str,
    reference: torch.Tensor,
) -> None:
    """Generic atom-to-system pooling should work with PET features."""

    data = _create_pet_test_data()

    featurizer = AtomisticFeaturizer(
        backbone=PETBackbone(
            model=DummyPET(),
        ),
        pooling=pooling,
        freeze=True,
    )

    output = featurizer(
        data
    )

    assert output.shape == (
        2,
        8,
    )

    assert torch.allclose(
        output,
        reference,
    )


def test_frozen_pet_stays_in_eval_mode() -> None:
    """A frozen PET model should remain in evaluation mode."""

    featurizer = AtomisticFeaturizer(
        backbone=PETBackbone(
            model=DummyPET(),
        ),
        pooling="mean",
        freeze=True,
    )

    featurizer.train()

    assert not featurizer.backbone.training
    assert not featurizer.backbone.model.training

    assert all(
        not parameter.requires_grad
        for parameter in featurizer.backbone.parameters()
    )


def test_trainable_pet_follows_featurizer_mode() -> None:
    """A non-frozen PET model should follow the wrapper training mode."""

    featurizer = AtomisticFeaturizer(
        backbone=PETBackbone(
            model=DummyPET(),
        ),
        pooling="mean",
        freeze=False,
    )

    featurizer.train()

    assert featurizer.backbone.training
    assert featurizer.backbone.model.training

    featurizer.eval()

    assert not featurizer.backbone.training
    assert not featurizer.backbone.model.training

    assert all(
        parameter.requires_grad
        for parameter in featurizer.backbone.parameters()
    )


def test_frozen_pet_preserves_position_gradients() -> None:
    """Freezing PET parameters must not detach atomic coordinates."""

    data = _create_pet_test_data()

    data["positions"].requires_grad_(
        True
    )

    featurizer = AtomisticFeaturizer(
        backbone=PETBackbone(
            model=DummyPET(),
        ),
        pooling="mean",
        freeze=True,
    )

    output = featurizer(
        data
    )

    output.sum().backward()

    assert data["positions"].grad is not None
    assert torch.isfinite(
        data["positions"].grad
    ).all()


def test_invalid_node_attribute_width() -> None:
    """The node encoding must follow PET's atomic-type table."""

    data = _create_pet_test_data()

    data["node_attrs"] = data[
        "node_attrs"
    ][:, :1]

    backbone = PETBackbone(
        model=DummyPET(),
    )

    with pytest.raises(
        ValueError,
        match="width of `node_attrs`",
    ):
        backbone(
            data
        )


def test_missing_graph_field() -> None:
    """Required graph fields should produce a clear error."""

    data = _create_pet_test_data()

    del data[
        "unit_shifts"
    ]

    backbone = PETBackbone(
        model=DummyPET(),
    )

    with pytest.raises(
        KeyError,
        match="missing required fields",
    ):
        backbone(
            data
        )
