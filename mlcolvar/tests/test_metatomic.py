"""Tests for system- and atom-level Metatomic model export."""
from typing import Dict

import pytest
import torch
from torch import nn

pytest.importorskip("metatensor.torch")
pytest.importorskip("metatomic.torch")

from metatensor.torch import Labels  # noqa: E402
from metatomic.torch import ModelOutput, NeighborListOptions  # noqa: E402

from mlcolvar.metatomic import create_metatomic_model, export_metatomic_model  # noqa: E402
from mlcolvar.metatomic.wrapper import _CVInferenceModel, _MetatomicCVWrapper  # noqa: E402


class DummyNetwork(nn.Module):
    def forward(self, data: Dict[str, torch.Tensor]) -> torch.Tensor:
        positions = data["positions"][:, :2]
        output = positions.new_zeros((data["ptr"].numel() - 1, 2))
        return output.index_add_(0, data["batch"], positions)


class DummyAtomNetwork(nn.Module):
    def forward(self, data: Dict[str, torch.Tensor]) -> torch.Tensor:
        return data["positions"][:, :2]


class DummyCVModel(nn.Module):
    def __init__(self, sample_kind="system") -> None:
        super().__init__()
        example = {
            "positions": torch.tensor([[0., 0., 0.], [1., 0., 0.]], dtype=torch.float64),
            "batch": torch.tensor([0, 0]),
            "ptr": torch.tensor([0, 2]),
        }
        network = DummyAtomNetwork() if sample_kind == "atom" else DummyNetwork()
        self.nn = torch.jit.trace(network.eval(), (example,), strict=False)
        self.n_cvs = 2
        self.output_kind = sample_kind
        self.register_buffer("atomic_numbers", torch.tensor([1, 8]))
        self.register_buffer("cutoff", torch.tensor(5.0, dtype=torch.float64))
        self.length_unit = "angstrom"


class _FakeNeighborList:
    def __init__(self, values: torch.Tensor) -> None:
        self.samples = type("Samples", (), {"values": values})()


class _FakeSystem:
    def __init__(self, types, positions, neighbors=None) -> None:
        if neighbors is None:
            neighbors = torch.empty((0, 5), dtype=torch.int32)
        self.types = torch.tensor(types, dtype=torch.long)
        self.positions = torch.tensor(positions, dtype=torch.float64).reshape(-1, 3)
        self.cell = torch.eye(3, dtype=torch.float64)
        self.pbc = torch.zeros(3, dtype=torch.bool)
        self._neighbor_list = _FakeNeighborList(neighbors)

    def __len__(self) -> int:
        return len(self.positions)

    def get_neighbor_list(self, options):
        return self._neighbor_list


def _neighbor_options():
    return NeighborListOptions(
        cutoff=5.0, full_list=True, strict=True, requestor="mlcolvar test"
    )


def _make_inference(network=None):
    return _CVInferenceModel(
        network=DummyNetwork() if network is None else network,
        postprocessing=nn.Identity(),
        atomic_numbers=torch.tensor([1, 8]),
        neighbor_options=_neighbor_options(),
    )


def _make_wrapper(sample_kind="system"):
    network = DummyAtomNetwork() if sample_kind == "atom" else DummyNetwork()
    return _MetatomicCVWrapper(_make_inference(network), 2, sample_kind)


def _make_systems():
    return [
        _FakeSystem(
            [1, 8], [[0., 0., 0.], [1., 0., 0.]],
            torch.tensor([[0, 1, 0, 0, 0], [1, 0, 0, 0, 0]], dtype=torch.int32),
        ),
        _FakeSystem([1], [[2., 1., 0.]]),
    ]


def test_systems_to_graph():
    graph = _make_inference()._systems_to_graph(_make_systems())
    torch.testing.assert_close(
        graph["positions"],
        torch.tensor([[0., 0., 0.], [1., 0., 0.], [2., 1., 0.]], dtype=torch.float64),
    )
    assert graph["batch"].tolist() == [0, 0, 1]
    assert graph["ptr"].tolist() == [0, 2, 3]
    assert graph["edge_index"].tolist() == [[0, 1], [1, 0]]


def test_metatomic_wrapper():
    block = _make_wrapper()(
        _make_systems(), {"feature": ModelOutput(sample_kind="system")}
    )["feature"].block()
    torch.testing.assert_close(
        block.values, torch.tensor([[1., 0.], [2., 1.]], dtype=torch.float64)
    )
    assert block.samples.names == ["system"]
    assert block.samples.values.tolist() == [[0], [1]]
    assert block.values.shape == (2, 2)
    assert len(block.properties) == 2


def test_metatomic_atom_outputs():
    block = _make_wrapper("atom")(
        _make_systems(), {"feature": ModelOutput(sample_kind="atom")}
    )["feature"].block()
    assert block.samples.names == ["system", "atom"]
    assert block.samples.values.tolist() == [[0, 0], [0, 1], [1, 0]]
    torch.testing.assert_close(
        block.values,
        torch.tensor([[0., 0.], [1., 0.], [2., 1.]], dtype=torch.float64),
    )


def test_metatomic_selected_atoms_gradients():
    systems = _make_systems()
    for system in systems:
        system.positions.requires_grad_(True)
    selected = Labels(
        names=["system", "atom"], values=torch.tensor([[0, 1], [1, 0]], dtype=torch.int32)
    )
    block = _make_wrapper("atom")(
        systems, {"feature": ModelOutput(sample_kind="atom")}, selected
    )["feature"].block()
    torch.testing.assert_close(block.samples.values, selected.values)
    torch.testing.assert_close(
        block.values, torch.tensor([[1., 0.], [2., 1.]], dtype=torch.float64)
    )
    block.values.sum().backward()
    torch.testing.assert_close(
        systems[0].positions.grad,
        torch.tensor([[0., 0., 0.], [1., 1., 0.]], dtype=torch.float64),
    )
    torch.testing.assert_close(
        systems[1].positions.grad, torch.tensor([[1., 1., 0.]], dtype=torch.float64)
    )


def test_metatomic_empty_atom_outputs():
    selected = Labels(
        names=["system", "atom"], values=torch.empty((0, 2), dtype=torch.int32)
    )
    block = _make_wrapper("atom")(
        _make_systems(), {"feature": ModelOutput(sample_kind="atom")}, selected
    )["feature"].block()
    assert block.values.shape == (0, 2)
    assert block.samples.values.shape == (0, 2)

    block = _make_wrapper("atom")(
        [_FakeSystem([], [])], {"feature": ModelOutput(sample_kind="atom")}
    )["feature"].block()
    assert block.values.shape == (0, 2)
    assert block.samples.names == ["system", "atom"]


def test_metatomic_atom_validation():
    wrapper = _make_wrapper("atom")
    with pytest.raises(ValueError, match="sample kind"):
        wrapper(_make_systems(), {"feature": ModelOutput(sample_kind="system")})

    for indices in ([[0, 2]], [[2, 0]], [[-1, 0]]):
        selected = Labels(
            names=["system", "atom"], values=torch.tensor(indices, dtype=torch.int32)
        )
        with pytest.raises(ValueError, match="out of range"):
            wrapper(_make_systems(), {"feature": ModelOutput(sample_kind="atom")}, selected)

    wrong_network = _MetatomicCVWrapper(_make_inference(DummyNetwork()), 2, "atom")
    with pytest.raises(ValueError, match="Output rows"):
        wrong_network(_make_systems(), {"feature": ModelOutput(sample_kind="atom")})

    with pytest.raises(ValueError, match="sample_kind"):
        create_metatomic_model(DummyCVModel(), sample_kind="invalid")


@pytest.mark.parametrize("sample_kind", ["system", "atom"])
def test_metatomic_export(tmp_path, sample_kind):
    path = tmp_path / f"{sample_kind}_model.pt"
    model = DummyCVModel(sample_kind)
    assert export_metatomic_model(model, path, supported_devices=["cpu"]) == path
    assert path.exists()
