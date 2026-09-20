from typing import Dict

import pytest
import torch
from torch import nn

pytest.importorskip("metatensor.torch")
pytest.importorskip("metatomic.torch")

from metatomic.torch import (  # noqa: E402
    ModelOutput,
    NeighborListOptions,
)

from mlcolvar.representation.metatomic import (  # noqa: E402
    CVInferenceModel,
    MetatomicCVWrapper,
)
from mlcolvar.representation.metatomic.export import (  # noqa: E402
    _get_neighbor_options,
    _infer_metadata,
    _prepare_network,
)


class DummyGraphNetwork(nn.Module):
    def forward(
        self,
        data: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        positions = data["positions"]
        batch = data["batch"]

        if batch.numel() == 0:
            return positions.new_zeros((0, 1))

        output = positions.new_zeros(
            (int(batch.max().item()) + 1, 1)
        )
        output.index_add_(
            0,
            batch,
            positions[:, :1],
        )
        return output


class DummyRepresentation(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.register_buffer(
            "atomic_numbers",
            torch.tensor([1, 6, 8]),
        )
        self.register_buffer(
            "cutoff",
            torch.tensor(
                5.0,
                dtype=torch.float64,
            ),
        )

        self.length_unit = "angstrom"


class DummyRepresentationNetwork(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.representation = DummyRepresentation()
        self.out_features = 2

        self.weight = nn.Parameter(
            torch.ones(
                1,
                dtype=torch.float64,
            )
        )


class DummyCVModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.nn = DummyRepresentationNetwork()
        self.n_cvs = 2


class _FakeSamples:
    def __init__(self, values: torch.Tensor) -> None:
        self.values = values


class _FakeNeighborList:
    def __init__(self, samples: torch.Tensor) -> None:
        self.samples = _FakeSamples(samples)


class _FakeSystem:
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
        self._neighbor_list = _FakeNeighborList(
            neighbors
        )

    def __len__(self) -> int:
        return self.positions.shape[0]

    def get_neighbor_list(
        self,
        options: NeighborListOptions,
    ) -> _FakeNeighborList:
        del options
        return self._neighbor_list


def _neighbor_options(
    cutoff: float = 3.0,
) -> NeighborListOptions:
    return NeighborListOptions(
        cutoff=cutoff,
        full_list=True,
        strict=True,
        requestor="mlcolvar test",
    )


def make_inference() -> CVInferenceModel:
    return CVInferenceModel(
        network=DummyGraphNetwork(),
        postprocessing=nn.Identity(),
        atomic_numbers=torch.tensor(
            [1, 8],
            dtype=torch.long,
        ),
        neighbor_options=_neighbor_options(),
    )


def make_system(
    types,
    positions,
    neighbors=None,
) -> _FakeSystem:
    if neighbors is None:
        neighbors = torch.empty(
            (0, 5),
            dtype=torch.int32,
        )

    return _FakeSystem(
        types=torch.tensor(
            types,
            dtype=torch.long,
        ),
        positions=torch.tensor(
            positions,
            dtype=torch.float64,
        ),
        cell=torch.eye(
            3,
            dtype=torch.float64,
        ),
        pbc=torch.zeros(
            3,
            dtype=torch.bool,
        ),
        neighbors=neighbors,
    )


def test_systems_to_graph() -> None:
    system_0 = make_system(
        [1, 8],
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ],
        torch.tensor(
            [
                [0, 1, 0, 0, 0],
                [1, 0, 0, 0, 0],
            ],
            dtype=torch.int32,
        ),
    )

    system_1 = make_system(
        [1],
        [
            [2.0, 0.0, 0.0],
        ],
    )

    graph = make_inference()._systems_to_graph(
        [system_0, system_1]
    )

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

    assert graph["edge_index"].tolist() == [
        [0, 1],
        [1, 0],
    ]
    assert graph["batch"].tolist() == [
        0,
        0,
        1,
    ]
    assert graph["ptr"].tolist() == [
        0,
        2,
        3,
    ]

    torch.testing.assert_close(
        graph["unit_shifts"],
        torch.zeros(
            2,
            3,
            dtype=torch.float64,
        ),
    )

    assert graph["cell"].shape == (
        2,
        3,
        3,
    )
    assert graph["pbc"].shape == (
        2,
        3,
    )


def test_neighbor_options_prefers_request() -> None:
    requested = _neighbor_options(
        cutoff=4.5
    )

    class Representation(nn.Module):
        def requested_neighbor_lists(
            self,
        ) -> list[NeighborListOptions]:
            return [requested]

    class Network(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.representation = (
                Representation()
            )

    resolved = _get_neighbor_options(
        Network(),
        interaction_range=8.0,
    )

    assert resolved is requested


def test_neighbor_options_uses_cutoff_fallback() -> None:
    class Representation(nn.Module):
        def __init__(self) -> None:
            super().__init__()

            self.register_buffer(
                "cutoff",
                torch.tensor(
                    5.5,
                    dtype=torch.float64,
                ),
            )
            self.full_neighbor_list = True

    class Network(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.representation = (
                Representation()
            )

    options = _get_neighbor_options(
        Network(),
        interaction_range=8.0,
    )

    assert options.cutoff == pytest.approx(
        5.5
    )
    assert options.full_list
    assert options.strict


def test_metatomic_metadata_is_inferred() -> None:
    metadata = _infer_metadata(
        DummyCVModel()
    )

    assert metadata["out_features"] == 2
    assert metadata["atomic_types"] == [
        1,
        6,
        8,
    ]
    assert metadata[
        "interaction_range"
    ] == pytest.approx(5.0)
    assert metadata["dtype"] == "float64"
    assert metadata[
        "length_unit"
    ] == "angstrom"


def test_prepare_network_finds_nested_representation() -> None:
    class Prepared(nn.Module):
        def __init__(self) -> None:
            super().__init__()

            self.register_buffer(
                "prepared",
                torch.tensor(False),
            )

        def prepare_for_torchscript(
            self,
        ) -> None:
            self.prepared.fill_(True)

    class Wrapper(nn.Module):
        def __init__(
            self,
            representation: nn.Module,
        ) -> None:
            super().__init__()

            self.representation = (
                representation
            )

    network = Wrapper(
        Wrapper(
            Prepared()
        )
    )

    prepared = _prepare_network(
        network
    )

    assert bool(
        prepared
        .representation
        .representation
        .prepared
        .item()
    )

    assert not bool(
        network
        .representation
        .representation
        .prepared
        .item()
    )


def test_metatomic_wrapper_empty_system() -> None:
    wrapper = MetatomicCVWrapper(
        model=make_inference(),
        out_features=2,
    )

    empty_system = make_system(
        [],
        [],
    )

    # Keep the expected position shape for an empty system.
    empty_system.positions = torch.empty(
        (0, 3),
        dtype=torch.float64,
    )

    result = wrapper(
        systems=[empty_system],
        outputs={
            "feature": ModelOutput(
                sample_kind="system"
            )
        },
        selected_atoms=None,
    )

    block = result["feature"].block()

    assert block.values.shape == (
        0,
        2,
    )
    assert block.values.dtype == (
        torch.float64
    )
    assert len(block.samples) == 0
    assert len(block.properties) == 2


def test_inference_model_is_torchscript_compatible() -> None:
    scripted = torch.jit.script(
        make_inference().eval()
    )

    assert isinstance(
        scripted,
        torch.jit.ScriptModule,
    )