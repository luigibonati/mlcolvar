from typing import Dict

import pytest
import torch
from torch import nn

pytest.importorskip("metatensor.torch")
pytest.importorskip("metatomic.torch")

from metatomic.torch import ModelOutput, NeighborListOptions  # noqa: E402

from mlcolvar.representation.metatomic import (  # noqa: E402
    CVInferenceModel,
    MetatomicCVWrapper,
    export_metatomic_model,
)


class DummyRepresentation(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.register_buffer(
            "atomic_numbers",
            torch.tensor([1, 8]),
        )
        self.register_buffer(
            "cutoff",
            torch.tensor(
                5.0,
                dtype=torch.float64,
            ),
        )

        self.length_unit = "angstrom"


class DummyNetwork(nn.Module):
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

    def forward(
        self,
        data: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        positions = data["positions"][:, :2]
        batch = data["batch"]
        ptr = data["ptr"]

        n_graphs = ptr.numel() - 1

        output = positions.new_zeros(
            (n_graphs, 2)
        )

        output.index_add_(
            0,
            batch,
            positions,
        )

        return output * self.weight


class DummyCVModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.nn = DummyNetwork()
        self.n_cvs = 2


class _FakeSamples:
    def __init__(
        self,
        values: torch.Tensor,
    ) -> None:
        self.values = values


class _FakeNeighborList:
    def __init__(
        self,
        samples: torch.Tensor,
    ) -> None:
        self.samples = _FakeSamples(samples)


class _FakeSystem:
    def __init__(
        self,
        types,
        positions,
        neighbors=None,
    ) -> None:
        if neighbors is None:
            neighbors = torch.empty(
                (0, 5),
                dtype=torch.int32,
            )

        self.types = torch.tensor(
            types,
            dtype=torch.long,
        )
        self.positions = torch.tensor(
            positions,
            dtype=torch.float64,
        )
        self.cell = torch.eye(
            3,
            dtype=torch.float64,
        )
        self.pbc = torch.zeros(
            3,
            dtype=torch.bool,
        )

        self._neighbor_list = _FakeNeighborList(
            neighbors
        )

    def __len__(self) -> int:
        return self.positions.shape[0]

    def get_neighbor_list(
        self,
        options: NeighborListOptions,
    ) -> _FakeNeighborList:
        return self._neighbor_list


def neighbor_options() -> NeighborListOptions:
    return NeighborListOptions(
        cutoff=5.0,
        full_list=True,
        strict=True,
        requestor="mlcolvar test",
    )


def make_inference() -> CVInferenceModel:
    return CVInferenceModel(
        network=DummyNetwork(),
        postprocessing=nn.Identity(),
        atomic_numbers=torch.tensor(
            [1, 8],
            dtype=torch.long,
        ),
        neighbor_options=neighbor_options(),
    )


def make_systems():
    return [
        _FakeSystem(
            types=[1, 8],
            positions=[
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
            ],
            neighbors=torch.tensor(
                [
                    [0, 1, 0, 0, 0],
                    [1, 0, 0, 0, 0],
                ],
                dtype=torch.int32,
            ),
        ),
        _FakeSystem(
            types=[1],
            positions=[
                [2.0, 1.0, 0.0],
            ],
        ),
    ]


def test_systems_to_graph() -> None:
    graph = make_inference()._systems_to_graph(
        make_systems()
    )

    torch.testing.assert_close(
        graph["positions"],
        torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [2.0, 1.0, 0.0],
            ],
            dtype=torch.float64,
        ),
    )

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
    assert graph["edge_index"].tolist() == [
        [0, 1],
        [1, 0],
    ]


def test_metatomic_wrapper() -> None:
    wrapper = MetatomicCVWrapper(
        model=make_inference(),
        out_features=2,
    )

    result = wrapper(
        systems=make_systems(),
        outputs={
            "feature": ModelOutput(
                sample_kind="system"
            )
        },
        selected_atoms=None,
    )

    block = result["feature"].block()

    torch.testing.assert_close(
        block.values,
        torch.tensor(
            [
                [1.0, 0.0],
                [2.0, 1.0],
            ],
            dtype=torch.float64,
        ),
    )

    assert block.values.shape == (2, 2)
    assert len(block.samples) == 2
    assert len(block.properties) == 2


def test_metatomic_export(
    tmp_path,
) -> None:
    path = tmp_path / "model.pt"

    result = export_metatomic_model(
        DummyCVModel(),
        path,
        supported_devices=["cpu"],
    )

    assert result == path
    assert path.exists()