from __future__ import annotations

from types import SimpleNamespace
from typing import Dict, List, Optional

import pytest
import torch
from torch import nn

import mlcolvar.featurization.atomistic.backbones.deepmd as deepmd_module

from mlcolvar.featurization.atomistic import (
    AtomisticFeaturizer,
    DeepMDBackbone,
)


class DummyDeepMDDescriptor(nn.Module):
    """Minimal DeePMD-like descriptor using float32 internally."""

    def __init__(self) -> None:
        super().__init__()
        self.prec = torch.float32
        self.precision = "float32"
        self.scale = nn.Parameter(torch.tensor(2.0, dtype=torch.float32))
        self.last_coord_dtype: Optional[torch.dtype] = None

    def get_rcut(self) -> float:
        return 2.5

    def get_sel(self) -> List[int]:
        return [8]

    def get_dim_out(self) -> int:
        return 4

    def mixed_types(self) -> bool:
        return True

    def forward(
        self,
        extended_coord: torch.Tensor,
        extended_atype: torch.Tensor,
        neighbor_list: torch.Tensor,
        mapping: Optional[torch.Tensor] = None,
    ):
        _ = neighbor_list
        _ = mapping

        self.last_coord_dtype = extended_coord.dtype

        if extended_coord.dtype != self.scale.dtype:
            raise RuntimeError(
                "Descriptor input dtype does not match its parameters."
            )

        type_feature = extended_atype.to(extended_coord.dtype).unsqueeze(-1)
        features = torch.cat([extended_coord, type_feature], dim=-1)

        return (features * self.scale, None, None, None, None)


class DummyDeepMDModel(nn.Module):
    """Minimal native DeePMD-like model."""

    def __init__(self) -> None:
        super().__init__()
        self.descriptor = DummyDeepMDDescriptor()

    def get_descriptor(self) -> nn.Module:
        return self.descriptor

    def get_type_map(self) -> List[str]:
        return ["H", "O"]


@pytest.fixture(autouse=True)
def fake_deepmd_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Provide the minimal DeePMD runtime required by unit tests."""

    monkeypatch.setattr(
        deepmd_module,
        "_DEEPMD_AVAILABLE",
        True,
    )
    monkeypatch.setattr(
        deepmd_module,
        "_DEEPMD_IMPORT_ERROR",
        None,
    )
    monkeypatch.setattr(
        deepmd_module,
        "ase_atomic_numbers",
        {"H": 1, "O": 8},
    )
    monkeypatch.setattr(
        deepmd_module,
        "deepmd_env",
        SimpleNamespace(
            GLOBAL_PT_FLOAT_PRECISION=torch.float32,
        ),
    )


@pytest.fixture
def fake_neighbor_builder(monkeypatch: pytest.MonkeyPatch):
    """Replace DeePMD neighbor construction with a deterministic stub."""

    calls = []

    def build(
        coord: torch.Tensor,
        atype: torch.Tensor,
        rcut: float,
        sel: List[int],
        mixed_types: bool,
        box: Optional[torch.Tensor],
    ):
        calls.append(
            {
                "coord_dtype": coord.dtype,
                "box_dtype": None if box is None else box.dtype,
                "rcut": float(rcut),
                "sel": list(sel),
                "mixed_types": bool(mixed_types),
            }
        )

        n_frames, n_atoms = coord.shape[:2]

        mapping = torch.arange(
            n_atoms,
            device=coord.device,
            dtype=torch.long,
        ).reshape(1, n_atoms).expand(n_frames, n_atoms)

        neighbor_list = torch.zeros(
            (n_frames, n_atoms, 1),
            device=coord.device,
            dtype=torch.long,
        )

        return coord, atype, mapping, neighbor_list

    monkeypatch.setattr(
        deepmd_module,
        "extend_input_and_build_neighbor_list",
        build,
    )

    return calls


def make_data(
    dtype: torch.dtype = torch.float64,
    periodic: bool = False,
) -> Dict[str, torch.Tensor]:
    """Create two two-atom systems encoded in DeePMD type-map order."""

    positions = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
        ],
        dtype=dtype,
    )

    node_attrs = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=dtype,
    )

    if periodic:
        cell = torch.eye(3, dtype=dtype).repeat(2, 1, 1) * 10.0
        pbc = torch.ones((2, 3), dtype=torch.bool)
    else:
        cell = torch.zeros((2, 3, 3), dtype=dtype)
        pbc = torch.zeros((2, 3), dtype=torch.bool)

    return {
        "positions": positions,
        "node_attrs": node_attrs,
        "batch": torch.tensor([0, 0, 1, 1], dtype=torch.long),
        "ptr": torch.tensor([0, 2, 4], dtype=torch.long),
        "cell": cell,
        "pbc": pbc,
    }


def test_deepmd_metadata() -> None:
    model = DummyDeepMDModel()
    backbone = DeepMDBackbone(model=model)

    assert backbone.model is model
    assert backbone.out_features == 4
    assert backbone.descriptor_dim == 4
    assert backbone.atomic_numbers.tolist() == [1, 8]
    assert backbone.descriptor_cutoff == pytest.approx(2.5)
    assert backbone.selection == [8]
    assert backbone.mixed_types
    assert not backbone.full_neighbor_list
    assert backbone._descriptor_dtype_reference.dtype == torch.float32


def test_deepmd_forward_preserves_external_dtype(
    fake_neighbor_builder,
) -> None:
    data = make_data(dtype=torch.float64)
    model = DummyDeepMDModel()
    backbone = DeepMDBackbone(model=model)

    # A surrounding float64 model must not convert the DeePMD descriptor.
    backbone.double()

    output = backbone(data)

    expected = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0, 2.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 4.0, 0.0, 2.0],
        ],
        dtype=torch.float64,
    )

    assert output.shape == (4, 4)
    assert output.dtype == torch.float64
    assert torch.allclose(output, expected)

    assert model.descriptor.scale.dtype == torch.float32
    assert model.descriptor.last_coord_dtype == torch.float32
    assert len(fake_neighbor_builder) == 2


def test_periodic_neighbor_list_uses_deepmd_precision(
    monkeypatch: pytest.MonkeyPatch,
    fake_neighbor_builder,
) -> None:
    monkeypatch.setattr(
        deepmd_module.deepmd_env,
        "GLOBAL_PT_FLOAT_PRECISION",
        torch.float64,
    )

    data = make_data(dtype=torch.float32, periodic=True)
    model = DummyDeepMDModel()
    backbone = DeepMDBackbone(model=model)

    output = backbone(data)

    assert output.dtype == torch.float32
    assert model.descriptor.last_coord_dtype == torch.float32
    assert len(fake_neighbor_builder) == 2

    for call in fake_neighbor_builder:
        assert call["coord_dtype"] == torch.float64
        assert call["box_dtype"] == torch.float64


def test_deepmd_featurizer_pooling_and_gradients(
    fake_neighbor_builder,
) -> None:
    data = make_data(dtype=torch.float64)
    data["positions"].requires_grad_(True)

    featurizer = AtomisticFeaturizer(
        backbone=DeepMDBackbone(model=DummyDeepMDModel()),
        pooling="mean",
        freeze=True,
    )

    output = featurizer(data)

    expected = torch.tensor(
        [
            [1.0, 0.0, 0.0, 1.0],
            [0.0, 2.0, 0.0, 1.0],
        ],
        dtype=torch.float64,
    )

    assert output.shape == (2, 4)
    assert torch.allclose(output, expected)

    output.sum().backward()

    assert data["positions"].grad is not None
    assert torch.isfinite(data["positions"].grad).all()

    assert all(
        not parameter.requires_grad
        for parameter in featurizer.backbone.parameters()
    )


def test_deepmd_rejects_long_range_cutoff() -> None:
    with pytest.raises(
        ValueError,
        match="does not support `long_range_cutoff`",
    ):
        DeepMDBackbone(
            model=DummyDeepMDModel(),
            long_range_cutoff=5.0,
        )


def test_deepmd_requires_optional_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Raise a clear error when DeePMD-kit is unavailable."""

    monkeypatch.setattr(
        deepmd_module,
        "_DEEPMD_AVAILABLE",
        False,
    )
    monkeypatch.setattr(
        deepmd_module,
        "_DEEPMD_IMPORT_ERROR",
        ImportError("deepmd"),
    )

    with pytest.raises(
        ImportError,
        match="requires DeePMD-kit",
    ):
        DeepMDBackbone(
            model=DummyDeepMDModel(),
        )