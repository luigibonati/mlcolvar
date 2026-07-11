from typing import Dict, List

import pytest
import torch
from torch import nn

pytest.importorskip("metatensor.torch")
pytest.importorskip("metatomic.torch")

from metatomic.torch import ModelOutput, NeighborListOptions, System

from mlcolvar.integrations.metatomic import (
    CVInferenceModel,
    MetatomicCVWrapper,
    _get_neighbor_options,
)


class DummyGraphNetwork(nn.Module):
    """Minimal mlcolvar-like atomistic model accepting a graph dictionary."""

    def forward(
        self,
        data: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        positions = data["positions"]
        batch = data["batch"]

        if batch.numel() == 0:
            return positions.new_zeros((0, 1))

        n_systems = int(batch.max().item()) + 1
        output = positions.new_zeros((n_systems, 1))
        output.index_add_(0, batch, positions[:, :1])
        return output


class _FakeSamples:
    def __init__(self, values: torch.Tensor) -> None:
        self.values = values


class _FakeNeighborList:
    def __init__(self, samples: torch.Tensor) -> None:
        self.samples = _FakeSamples(samples)


class _FakeSystem:
    """Small eager-mode stand-in for metatomic.torch.System."""

    def __init__(
        self,
        types: torch.Tensor,
        positions: torch.Tensor,
        cell: torch.Tensor,
        pbc: torch.Tensor,
        neighbors: torch.Tensor,
    ) -> None:
        self.types = types
        self.positions = positions
        self.cell = cell
        self.pbc = pbc
        self._neighbor_list = _FakeNeighborList(neighbors)

    def __len__(self) -> int:
        return self.positions.shape[0]

    def get_neighbor_list(
        self,
        options: NeighborListOptions,
    ) -> _FakeNeighborList:
        return self._neighbor_list


def _neighbor_options(cutoff: float = 3.0) -> NeighborListOptions:
    return NeighborListOptions(
        cutoff=cutoff,
        full_list=True,
        strict=True,
        requestor="mlcolvar test",
    )


def test_systems_to_graph() -> None:
    """Metatomic systems are converted to the expected mlcolvar graph."""
    model = CVInferenceModel(
        network=DummyGraphNetwork(),
        postprocessing=nn.Identity(),
        atomic_numbers=torch.tensor([1, 8]),
        neighbor_options=_neighbor_options(),
    )

    system_0 = _FakeSystem(
        types=torch.tensor([1, 8], dtype=torch.long),
        positions=torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            dtype=torch.float64,
        ),
        cell=torch.eye(3, dtype=torch.float64),
        pbc=torch.tensor([False, False, False]),
        neighbors=torch.tensor(
            [
                [0, 1, 0, 0, 0],
                [1, 0, 0, 0, 0],
            ],
            dtype=torch.int32,
        ),
    )

    system_1 = _FakeSystem(
        types=torch.tensor([1], dtype=torch.long),
        positions=torch.tensor(
            [[2.0, 0.0, 0.0]],
            dtype=torch.float64,
        ),
        cell=torch.eye(3, dtype=torch.float64),
        pbc=torch.tensor([False, False, False]),
        neighbors=torch.empty((0, 5), dtype=torch.int32),
    )

    graph = model._systems_to_graph([system_0, system_1])

    torch.testing.assert_close(
        graph["positions"],
        torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
            ],
            dtype=torch.float64,
        ),
    )
    torch.testing.assert_close(
        graph["node_attrs"],
        torch.tensor(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 0.0],
            ],
            dtype=torch.float64,
        ),
    )
    torch.testing.assert_close(
        graph["edge_index"],
        torch.tensor(
            [[0, 1], [1, 0]],
            dtype=torch.long,
        ),
    )
    torch.testing.assert_close(
        graph["batch"],
        torch.tensor([0, 0, 1], dtype=torch.long),
    )
    torch.testing.assert_close(
        graph["ptr"],
        torch.tensor([0, 2, 3], dtype=torch.long),
    )
    assert graph["unit_shifts"].shape == (2, 3)
    assert graph["cell"].shape == (2, 3, 3)
    assert graph["pbc"].shape == (2, 3)


def test_neighbor_options_resolution() -> None:
    """Standard requests are preferred, with cutoff fallback when absent."""
    requested = _neighbor_options(cutoff=4.5)

    class BackboneWithRequest(nn.Module):
        def requested_neighbor_lists(self) -> List[NeighborListOptions]:
            return [requested]

    class Featurizer(nn.Module):
        def __init__(self, backbone: nn.Module) -> None:
            super().__init__()
            self.backbone = backbone

    class Network(nn.Module):
        def __init__(self, backbone: nn.Module) -> None:
            super().__init__()
            self.featurizer = Featurizer(backbone)

    resolved = _get_neighbor_options(
        network=Network(BackboneWithRequest()),
        interaction_range=8.0,
    )
    assert resolved is requested

    class BackboneWithCutoff(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.register_buffer(
                "cutoff",
                torch.tensor(5.5, dtype=torch.float64),
            )

    fallback = _get_neighbor_options(
        network=Network(BackboneWithCutoff()),
        interaction_range=8.0,
    )
    assert fallback.cutoff == pytest.approx(5.5)


def test_metatomic_wrapper_empty_system() -> None:
    """PLUMED's empty-system probe returns zero samples and fixed properties."""
    inference = CVInferenceModel(
        network=DummyGraphNetwork(),
        postprocessing=nn.Identity(),
        atomic_numbers=torch.tensor([1, 8]),
        neighbor_options=_neighbor_options(),
    )
    wrapper = MetatomicCVWrapper(
        model=inference,
        out_features=2,
    )

    empty_system = _FakeSystem(
        types=torch.empty(0, dtype=torch.long),
        positions=torch.empty((0, 3), dtype=torch.float64),
        cell=torch.eye(3, dtype=torch.float64),
        pbc=torch.tensor([False, False, False]),
        neighbors=torch.empty((0, 5), dtype=torch.int32),
    )

    result = wrapper(
        systems=[empty_system],
        outputs={"feature": ModelOutput(sample_kind="system")},
        selected_atoms=None,
    )

    block = result["feature"].block()
    assert block.values.shape == (0, 2)
    assert len(block.samples) == 0
    assert len(block.properties) == 2


def test_inference_model_is_torchscript_compatible() -> None:
    """The System-to-graph adapter can be compiled by TorchScript."""
    inference = CVInferenceModel(
        network=DummyGraphNetwork(),
        postprocessing=nn.Identity(),
        atomic_numbers=torch.tensor([1, 8]),
        neighbor_options=_neighbor_options(),
    ).eval()

    scripted = torch.jit.script(inference)

    assert isinstance(scripted, torch.jit.ScriptModule)
