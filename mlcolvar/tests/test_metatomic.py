from typing import Dict

import pytest
import torch
from torch import nn

pytest.importorskip("metatensor.torch")
pytest.importorskip("metatomic.torch")

from metatomic.torch import ModelOutput, NeighborListOptions  # noqa: E402

from mlcolvar.metatomic import export_metatomic_model  # noqa: E402
from mlcolvar.metatomic.wrapper import (  # noqa: E402
    _CVInferenceModel,
    _MetatomicCVWrapper,
)


class DummyNetwork(nn.Module):
    def forward(self, data: Dict[str, torch.Tensor]) -> torch.Tensor:
        positions = data["positions"][:, :2]
        batch = data["batch"]
        output = positions.new_zeros((data["ptr"].numel() - 1, 2))
        output.index_add_(0, batch, positions)
        return output


class DummyCVModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        example = {
            "positions": torch.tensor(
                [[0., 0., 0.], [1., 0., 0.]],
                dtype=torch.float64,
            ),
            "batch": torch.tensor([0, 0]),
            "ptr": torch.tensor([0, 2]),
        }
        self.nn = torch.jit.trace(
            DummyNetwork().eval(),
            (example,),
            strict=False,
        )
        self.n_cvs = 2
        self.register_buffer("atomic_numbers", torch.tensor([1, 8]))
        self.register_buffer(
            "cutoff",
            torch.tensor(5.0, dtype=torch.float64),
        )
        self.length_unit = "angstrom"


class _FakeNeighborList:
    def __init__(self, values: torch.Tensor) -> None:
        self.samples = type("Samples", (), {"values": values})()


class _FakeSystem:
    def __init__(self, types, positions, neighbors=None) -> None:
        if neighbors is None:
            neighbors = torch.empty((0, 5), dtype=torch.int32)
        self.types = torch.tensor(types, dtype=torch.long)
        self.positions = torch.tensor(positions, dtype=torch.float64)
        self.cell = torch.eye(3, dtype=torch.float64)
        self.pbc = torch.zeros(3, dtype=torch.bool)
        self._neighbor_list = _FakeNeighborList(neighbors)

    def __len__(self) -> int:
        return len(self.positions)

    def get_neighbor_list(self, options):
        return self._neighbor_list


def _neighbor_options():
    return NeighborListOptions(
        cutoff=5.0,
        full_list=True,
        strict=True,
        requestor="mlcolvar test",
    )


def _make_inference():
    return _CVInferenceModel(
        network=DummyNetwork(),
        postprocessing=nn.Identity(),
        atomic_numbers=torch.tensor([1, 8]),
        neighbor_options=_neighbor_options(),
    )


def _make_systems():
    return [
        _FakeSystem(
            [1, 8],
            [[0., 0., 0.], [1., 0., 0.]],
            torch.tensor(
                [[0, 1, 0, 0, 0], [1, 0, 0, 0, 0]],
                dtype=torch.int32,
            ),
        ),
        _FakeSystem([1], [[2., 1., 0.]]),
    ]


def test_systems_to_graph():
    graph = _make_inference()._systems_to_graph(_make_systems())
    torch.testing.assert_close(
        graph["positions"],
        torch.tensor(
            [[0., 0., 0.], [1., 0., 0.], [2., 1., 0.]],
            dtype=torch.float64,
        ),
    )
    assert graph["batch"].tolist() == [0, 0, 1]
    assert graph["ptr"].tolist() == [0, 2, 3]
    assert graph["edge_index"].tolist() == [[0, 1], [1, 0]]


def test_metatomic_wrapper():
    result = _MetatomicCVWrapper(
        model=_make_inference(),
        out_features=2,
    )(
        systems=_make_systems(),
        outputs={"feature": ModelOutput(sample_kind="system")},
        selected_atoms=None,
    )
    block = result["feature"].block()
    torch.testing.assert_close(
        block.values,
        torch.tensor([[1., 0.], [2., 1.]], dtype=torch.float64),
    )
    assert block.values.shape == (2, 2)
    assert len(block.samples) == 2
    assert len(block.properties) == 2


def test_metatomic_export(tmp_path):
    path = tmp_path / "model.pt"
    assert export_metatomic_model(
        DummyCVModel(),
        path,
        supported_devices=["cpu"],
    ) == path
    assert path.exists()