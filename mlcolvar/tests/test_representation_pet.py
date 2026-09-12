from __future__ import annotations

from typing import Dict, List, Optional

import pytest
import torch
from torch import nn

# Import the PET adapter first: it installs the PET TorchScript compatibility
# patch before metatensor/metatomic internals are imported.
pytest.importorskip("mlcolvar.representation.adapters.pet")

metatensor_torch = pytest.importorskip("metatensor.torch")
metatomic_torch = pytest.importorskip("metatomic.torch")

Labels = metatensor_torch.Labels
TensorBlock = metatensor_torch.TensorBlock
TensorMap = metatensor_torch.TensorMap

ModelOutput = metatomic_torch.ModelOutput
NeighborListOptions = metatomic_torch.NeighborListOptions
System = metatomic_torch.System

from mlcolvar.representation import (  # noqa: E402
    PETRepresentation,
    PoolReducer,
    RepresentationModel,
)


class DummyPET(nn.Module):
    """Minimal PET-like model using float32 internally."""

    def __init__(self) -> None:
        super().__init__()

        self.atomic_types = [1, 8]
        self.cutoff = 2.5

        self.d_node = 2
        self.d_pet = 2
        self.num_readout_layers = 2

        self.scale = nn.Parameter(
            torch.tensor(1.0, dtype=torch.float32)
        )
        self.last_position_dtype: Optional[torch.dtype] = None

        self._neighbor_options = NeighborListOptions(
            cutoff=self.cutoff,
            full_list=True,
            strict=True,
        )

    def supported_outputs(self) -> Dict[str, ModelOutput]:
        return {
            "feature": ModelOutput(
                sample_kind="atom",
            )
        }

    def requested_neighbor_lists(self) -> List[NeighborListOptions]:
        return [self._neighbor_options]

    def forward(
        self,
        systems: List[System],
        outputs: Dict[str, ModelOutput],
        selected_atoms: Optional[Labels] = None,
    ) -> Dict[str, TensorMap]:
        assert "feature" in outputs
        assert selected_atoms is None

        values_list: List[torch.Tensor] = []
        sample_rows: List[List[int]] = []

        for system_index, system in enumerate(systems):
            _ = system.get_neighbor_list(self._neighbor_options)

            positions = system.positions
            self.last_position_dtype = positions.dtype

            if positions.dtype != self.scale.dtype:
                raise RuntimeError(
                    "PET input dtype does not match model parameters."
                )

            atom_types = system.types.to(
                dtype=positions.dtype,
            ).reshape(-1, 1)

            base_features = torch.cat(
                [positions, atom_types],
                dim=1,
            )

            # Mimic two PET readout layers.
            features = torch.cat(
                [base_features, base_features],
                dim=1,
            )

            values_list.append(features * self.scale)

            for atom_index in range(positions.size(0)):
                sample_rows.append(
                    [system_index, atom_index]
                )

        values = torch.cat(values_list, dim=0)

        samples = Labels(
            names=["system", "atom"],
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
    """Minimal wrapper exposing the native PET model through `.model`."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model


def make_data(
    dtype: torch.dtype = torch.float64,
) -> Dict[str, torch.Tensor]:
    """Create two non-periodic two-atom systems."""

    return {
        "positions": torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 2.0, 0.0],
            ],
            dtype=dtype,
        ),
        "node_attrs": torch.tensor(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 0.0],
                [0.0, 1.0],
            ],
            dtype=dtype,
        ),
        "edge_index": torch.tensor(
            [
                [0, 1, 2, 3],
                [1, 0, 3, 2],
            ],
            dtype=torch.long,
        ),
        "unit_shifts": torch.zeros(
            (4, 3),
            dtype=dtype,
        ),
        "batch": torch.tensor(
            [0, 0, 1, 1],
            dtype=torch.long,
        ),
        "ptr": torch.tensor(
            [0, 2, 4],
            dtype=torch.long,
        ),
        "cell": torch.zeros(
            (2, 3, 3),
            dtype=dtype,
        ),
        "pbc": torch.zeros(
            (2, 3),
            dtype=torch.bool,
        ),
    }


def make_representation() -> PETRepresentation:
    return PETRepresentation(
        model=DummyPET(),
        freeze=True,
    )


def test_pet_representation_metadata_and_wrapper() -> None:
    native_pet = DummyPET()

    representation = PETRepresentation(
        model=DummyPETWrapper(native_pet),
        freeze=True,
    )

    assert representation.model is native_pet

    assert representation.input_kind == "graph"
    assert representation.output_kind == "atom"
    assert representation.in_features is None
    assert representation.out_features == 8
    assert representation.freeze
    assert representation.full_neighbor_list

    assert representation.atomic_numbers.tolist() == [1, 8]
    assert representation.cutoff.item() == pytest.approx(2.5)

    assert representation.neighbor_cutoff == pytest.approx(2.5)
    assert representation.neighbor_full_list
    assert representation.neighbor_strict

    assert (
        representation._model_dtype_reference.dtype
        == torch.float32
    )

    assert all(
        not parameter.requires_grad
        for parameter in representation.parameters()
    )


def test_pet_neighbor_list_conversion() -> None:
    data = make_data()
    representation = make_representation()

    neighbors = representation._build_neighbor_list(
        edge_index=data["edge_index"],
        unit_shifts=data["unit_shifts"],
        long_range_mask=None,
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
    torch.testing.assert_close(
        neighbors.values.squeeze(-1),
        expected_vectors,
    )


def test_pet_forward_preserves_external_dtype() -> None:
    data = make_data(dtype=torch.float64)
    native_pet = DummyPET()

    representation = PETRepresentation(
        model=native_pet,
        freeze=True,
    )

    representation.double()
    output = representation(data)

    expected = torch.tensor(
        [
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, 8.0, 1.0, 0.0, 0.0, 8.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            [0.0, 2.0, 0.0, 8.0, 0.0, 2.0, 0.0, 8.0],
        ],
        dtype=torch.float64,
    )

    assert output.shape == (4, 8)
    assert output.dtype == torch.float64
    torch.testing.assert_close(output, expected)

    assert native_pet.scale.dtype == torch.float32
    assert native_pet.last_position_dtype == torch.float32


def test_pet_representation_preserves_coordinate_gradients() -> None:
    data = make_data(dtype=torch.float64)
    data["positions"].requires_grad_(True)

    representation = make_representation()
    output = representation(data)

    output.sum().backward()

    gradient = data["positions"].grad
    assert gradient is not None
    assert gradient.shape == data["positions"].shape
    assert torch.isfinite(gradient).all()
    assert torch.count_nonzero(gradient) > 0

    assert all(
        not parameter.requires_grad
        for parameter in representation.parameters()
    )

    representation.train()
    assert not representation.training
    assert not representation.model.training


def test_pet_mean_pooling() -> None:
    data = make_data(dtype=torch.float64)
    representation = make_representation()

    reducer = PoolReducer(
        in_features=representation.out_features,
        pooling="mean",
    )

    output = reducer(
        representation(data),
        data,
    )

    expected = torch.tensor(
        [
            [0.5, 0.0, 0.0, 4.5, 0.5, 0.0, 0.0, 4.5],
            [0.0, 1.0, 0.0, 4.5, 0.0, 1.0, 0.0, 4.5],
        ],
        dtype=torch.float64,
    )

    assert output.shape == (2, 8)
    torch.testing.assert_close(output, expected)


def test_pet_representation_model_default_pooling() -> None:
    representation = make_representation()

    model = RepresentationModel(
        representation,
        n_out=1,
        hidden_layers=(),
    )

    output = model(make_data())

    assert model.representation is representation
    assert isinstance(model.pre_head, PoolReducer)
    assert output.shape == (2, 1)

    assert all(
        not parameter.requires_grad
        for parameter in representation.parameters()
    )
    assert any(
        parameter.requires_grad
        for parameter in model.head.parameters()
    )


def test_invalid_pet_model() -> None:
    with pytest.raises(
        ValueError,
        match="Could not find a native metatrain PET model",
    ):
        PETRepresentation(
            model=nn.Linear(2, 2),
        )
