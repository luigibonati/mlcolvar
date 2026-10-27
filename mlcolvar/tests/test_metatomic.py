from typing import Dict

import pytest
import torch
from torch import nn

pytest.importorskip("metatensor.torch")
pytest.importorskip("metatomic.torch")

from metatensor.torch import Labels  # noqa: E402
from metatomic.torch import ModelOutput, NeighborListOptions  # noqa: E402
from mlcolvar.metatomic import create_metatomic_model, export_metatomic_model  # noqa: E402
from mlcolvar.metatomic.export import _get_neighbor_options, _resolve_interaction_range  # noqa: E402
from mlcolvar.core.nn.graph import SchNetModel  # noqa: E402
from mlcolvar.representation import MACERepresentation, MLColvarRepresentation  # noqa: E402
from mlcolvar.metatomic.wrapper import _CVInferenceModel, _MetatomicCVWrapper  # noqa: E402


class DummyNetwork(nn.Module):
    def forward(self, data: Dict[str, torch.Tensor]) -> torch.Tensor:
        positions = data["positions"][:, :2]
        output = positions.new_zeros((data["ptr"].numel() - 1, 2))
        return output.index_add_(0, data["batch"], positions)


class DummyAtomNetwork(nn.Module):
    def forward(self, data: Dict[str, torch.Tensor]) -> torch.Tensor:
        return data["positions"][:, :2]


class EdgeDistanceNetwork(nn.Module):
    def forward(self, data: Dict[str, torch.Tensor]) -> torch.Tensor:
        src, dst = data["edge_index"]
        vectors = data["positions"][dst] - data["positions"][src] + data["shifts"]
        return vectors.norm(dim=1).sum().reshape(1, 1)


class DummyCVModel(nn.Module):
    def __init__(self, sample_kind="system") -> None:
        super().__init__()
        example = {
            "positions": torch.tensor([[0., 0., 0.], [1., 0., 0.]], dtype=torch.float64),
            "batch": torch.tensor([0, 0]), "ptr": torch.tensor([0, 2]),
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
    def __init__(self, types, positions, neighbors=None, cell=None, pbc=None) -> None:
        self.types = torch.tensor(types, dtype=torch.long)
        self.positions = torch.tensor(positions, dtype=torch.float64).reshape(-1, 3)
        self.cell = torch.eye(3, dtype=torch.float64) if cell is None else torch.tensor(cell, dtype=torch.float64)
        self.pbc = torch.zeros(3, dtype=torch.bool) if pbc is None else torch.tensor(pbc, dtype=torch.bool)
        self.neighbors = neighbors

    def __len__(self) -> int:
        return len(self.positions)

    def get_neighbor_list(self, options):
        if isinstance(self.neighbors, dict):
            values = self.neighbors[float(options.cutoff)]
        elif self.neighbors is not None:
            values = self.neighbors
        else:
            # Non-periodic dynamic fake lists. Periodic edges are supplied explicitly in tests.
            distances = torch.cdist(self.positions.detach(), self.positions.detach())
            keep = (distances < options.cutoff) & ~torch.eye(len(self), dtype=torch.bool)
            src, dst = torch.where(keep)
            shifts = torch.zeros((len(src), 3), dtype=torch.long)
            values = torch.cat((src[:, None], dst[:, None], shifts), dim=1).to(torch.int32)
        return _FakeNeighborList(values)


def _neighbor_options(cutoff=5.0):
    return NeighborListOptions(cutoff=cutoff, full_list=True, strict=True, requestor="mlcolvar test")


def _make_inference(network=None, *, short_cutoff=5.0, long_cutoff=None, graph_mode="merged", **kwargs):
    return _CVInferenceModel(
        network=DummyNetwork() if network is None else network,
        postprocessing=nn.Identity(), atomic_numbers=torch.tensor([1, 8]),
        neighbor_options=_neighbor_options(short_cutoff),
        long_neighbor_options=_neighbor_options(long_cutoff) if long_cutoff is not None else None,
        short_cutoff=short_cutoff, long_cutoff=long_cutoff,
        graph_mode=graph_mode, **kwargs,
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
    torch.testing.assert_close(block.values, torch.tensor([[1., 0.], [2., 1.]], dtype=torch.float64))
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
        block.values, torch.tensor([[0., 0.], [1., 0.], [2., 1.]], dtype=torch.float64),
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
    torch.testing.assert_close(block.values, torch.tensor([[1., 0.], [2., 1.]], dtype=torch.float64))
    block.values.sum().backward()
    torch.testing.assert_close(
        systems[0].positions.grad,
        torch.tensor([[0., 0., 0.], [1., 1., 0.]], dtype=torch.float64),
    )
    torch.testing.assert_close(
        systems[1].positions.grad, torch.tensor([[1., 1., 0.]], dtype=torch.float64),
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
        selected = Labels(names=["system", "atom"], values=torch.tensor(indices, dtype=torch.int32))
        with pytest.raises(ValueError, match="out of range"):
            wrapper(_make_systems(), {"feature": ModelOutput(sample_kind="atom")}, selected)
    wrong_network = _MetatomicCVWrapper(_make_inference(DummyNetwork()), 2, "atom")
    with pytest.raises(ValueError, match="Output rows"):
        wrong_network(_make_systems(), {"feature": ModelOutput(sample_kind="atom")})
    with pytest.raises(ValueError, match="sample_kind"):
        create_metatomic_model(DummyCVModel(), sample_kind="invalid")


def test_dual_cutoff_edges_and_masks():
    systems = [
        _FakeSystem([1, 8, 8], [[0, 0, 0], [0.8, 0, 0], [2, 0, 0]]),
        _FakeSystem([1, 8], [[0, 0, 0], [0.9, 0, 0]]),
    ]
    model = _make_inference(short_cutoff=1.0, long_cutoff=4.0, subsystem_indices=[0, 1])
    graph = model._systems_to_graph(systems)
    assert len(model.requested_neighbor_lists()) == 2
    assert graph["ptr"].tolist() == [0, 3, 5]
    torch.testing.assert_close(
        graph["edge_index_short"], torch.tensor([[0, 1, 3, 4], [1, 0, 4, 3]])
    )
    torch.testing.assert_close(graph["edge_index_long"], graph["edge_index_short"])
    assert graph["edge_masks_lr"].flatten().tolist() == [False] * 4 + [True] * 4
    torch.testing.assert_close(
        graph["edge_index"], torch.cat((graph["edge_index_short"], graph["edge_index_long"]), dim=1)
    )


def test_dynamic_environment_selection():
    system = _FakeSystem([1, 8, 8, 8], [[0, 0, 0], [0.8, 0, 0], [2, 0, 0], [5, 0, 0]])
    model = _make_inference(
        short_cutoff=1.0, long_cutoff=4.0, buffer=1.5,
        system_indices=[0], environment_indices=[2], subsystem_indices=[0, 2],
    )
    graph = model._systems_to_graph([system])
    assert graph["system_masks"].flatten().tolist() == [True, False, False, False]
    assert graph["n_system"].tolist() == [[1.0]]
    assert graph["n_env"].tolist() == [[3.0]]
    assert graph["edge_index_short"].shape == (2, 0)
    assert graph["edge_index_long"].shape == (2, 2)
    with torch.no_grad():
        system.positions[2, 0] = 0.9
    updated = model._systems_to_graph([system])
    assert updated["edge_index_short"].tolist() == [[0, 2], [2, 0]]


def test_variable_atom_counts_with_species_selection():
    systems = [
        _FakeSystem([1, 8, 1], [[0, 0, 0], [1, 0, 0], [2, 0, 0]]),
        _FakeSystem([1, 8], [[0, 0, 0], [1, 0, 0]]),
    ]
    graph = _make_inference(
        short_cutoff=1.0, long_cutoff=4.0, subsystem_atomic_numbers=[1]
    )._systems_to_graph(systems)
    assert graph["subsystem_masks"].flatten().tolist() == [True, False, True, True, False]
    assert graph["ptr"].tolist() == [0, 3, 5]
    assert graph["edge_index_long"].tolist() == [[0, 2], [2, 0]]


def test_separate_neighbor_lists():
    system = _FakeSystem([1, 8, 8], [[0, 0, 0], [0.8, 0, 0], [2, 0, 0]])
    model = _make_inference(short_cutoff=1.0, long_cutoff=4.0, graph_mode="separate")
    graph = model._systems_to_graph([system])
    assert "edge_masks_lr" not in graph
    assert graph["edge_index"].equal(graph["edge_index_short"])
    assert graph["edge_index_short"].shape == (2, 2)
    assert graph["edge_index_long"].shape == (2, 6)
    system.positions.requires_grad_(True)
    model.network = DummyAtomNetwork()
    model([system]).sum().backward()
    assert system.positions.grad is not None


def test_periodic_shifts_and_gradients():
    neighbors = torch.tensor(
        [[0, 1, -1, 0, 0], [1, 0, 1, 0, 0]], dtype=torch.int32
    )
    system = _FakeSystem(
        [1, 8], [[0.1, 0, 0], [4.9, 0, 0]], neighbors=neighbors,
        cell=[[5, 0, 0], [0, 5, 0], [0, 0, 5]], pbc=[True, False, False],
    )
    model = _make_inference(network=EdgeDistanceNetwork(), short_cutoff=1.0)
    graph = model._systems_to_graph([system])
    torch.testing.assert_close(
        graph["shifts"], torch.tensor([[-5., 0., 0.], [5., 0., 0.]], dtype=torch.float64)
    )
    system.positions.requires_grad_(True)
    model([system]).sum().backward()
    torch.testing.assert_close(
        system.positions.grad, torch.tensor([[2., 0., 0.], [-2., 0., 0.]], dtype=torch.float64)
    )


def test_neighbor_request_reuse_and_validation():
    class NativeNetwork(nn.Module):
        def __init__(self):
            super().__init__()
            self.neighbor_options = _neighbor_options(1.0)

    short, long = _get_neighbor_options(NativeNetwork(), 1.0, 4.0, buffer=1.5)
    assert short.cutoff == 1.0 and long.cutoff == 4.0
    with pytest.raises(ValueError, match="subsystem"):
        _make_inference(short_cutoff=1.0, long_cutoff=4.0)
    with pytest.raises(ValueError, match="exceed"):
        _make_inference(
            short_cutoff=1.0, long_cutoff=4.0, subsystem_indices=[4]
        )._systems_to_graph([_FakeSystem([1, 8], [[0, 0, 0], [0.8, 0, 0]])])
    with pytest.raises(ValueError, match="unique"):
        _make_inference(short_cutoff=1.0, long_cutoff=4.0, subsystem_indices=[0, 0])


@pytest.mark.parametrize("sample_kind", ["system", "atom"])
def test_metatomic_export(tmp_path, sample_kind):
    path = tmp_path / f"{sample_kind}_model.pt"
    assert export_metatomic_model(
        DummyCVModel(sample_kind), path, supported_devices=["cpu"]
    ) == path
    assert path.exists()


def test_metatomic_dual_neighbor_export(tmp_path):
    path = tmp_path / "dual_cutoff_model.pt"
    assert export_metatomic_model(
        DummyCVModel(), path, long_range_cutoff=7.0,
        subsystem_indices=[0, 1], supported_devices=["cpu"],
    ) == path
    assert path.exists()


def test_interaction_range_inference():
    schnet = SchNetModel(
        n_out=2, dataset_for_initialization=None, cutoff=5.0,
        atomic_numbers=[1, 8], n_layers=2,
    )
    assert _resolve_interaction_range(schnet, None, 5.0, None, 0.0) == 10.0
    assert _resolve_interaction_range(schnet, None, 5.0, 8.0, 0.0) == 16.0

    class PretrainedCV(nn.Module):
        def __init__(self):
            super().__init__()
            self.nn = schnet

    representation = MLColvarRepresentation(PretrainedCV())
    assert _resolve_interaction_range(representation, None, 5.0, 8.0, 0.0) == 16.0

    class PretrainedMACE(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("atomic_numbers", torch.tensor([1, 8]))
            self.register_buffer("r_max", torch.tensor(5.0))
            self.register_buffer("num_interactions", torch.tensor(3))

    mace = MACERepresentation(
        PretrainedMACE(), num_layers=2, num_features=2, l_max=1,
    )
    assert _resolve_interaction_range(mace, None, 5.0, None, 0.0) == 15.0

    with pytest.raises(ValueError, match="interaction_range"):
        _resolve_interaction_range(schnet, 8.0, 5.0, None, 0.0)

    # Unknown networks are conservatively treated as having non-local reach.
    assert _resolve_interaction_range(DummyNetwork(), None, 5.0, None, 0.0) == float("inf")
    assert _resolve_interaction_range(DummyNetwork(), 5.0, 5.0, None, 0.0) == 5.0
