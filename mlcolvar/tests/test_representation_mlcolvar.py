import torch
from torch import nn

from mlcolvar.core import BaseGNN, FeedForward
from mlcolvar.cvs import RegressionCV
from mlcolvar.representation import (
    MLColvarRepresentation,
    Representation,
    export_representation_torchscript,
)


class AddCellPreprocessing(nn.Module):
    def forward(
        self,
        x: torch.Tensor,
        cell: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = x + 1.0
        return x if cell is None else x + cell.reshape(())


class ScaleNormalization(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 2.0 * x


class DummyDescriptorCV(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.in_features = 3
        self.preprocessing = AddCellPreprocessing()
        self.norm_in = ScaleNormalization()
        self.nn = nn.Linear(3, 2, bias=False)

        with torch.no_grad():
            self.nn.weight.copy_(
                torch.tensor(
                    [
                        [1.0, 0.0, 0.0],
                        [0.0, 1.0, 1.0],
                    ]
                )
            )


class GraphShiftPreprocessing(nn.Module):
    def forward(
        self,
        data: dict[str, torch.Tensor],
        cell: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        output = dict(data)
        shift = 1.0 if cell is None else 1.0 + cell.reshape(())
        output["positions"] = data["positions"] + shift
        return output


class DummyGraphEncoder(BaseGNN):
    def __init__(
        self,
        pooling_operation: str | None = "mean",
    ) -> None:
        super().__init__(
            n_out=2,
            dataset_for_initialization=None,
            pooling_operation=pooling_operation,
            cutoff=4.0,
            buffer=0.5,
            long_range_cutoff=-1.0,
            atomic_numbers=[1, 6, 8],
        )
        self.scale = nn.Parameter(torch.tensor(2.0))

    def forward(
        self,
        data: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        return self.pooling(
            data["positions"][:, :2] * self.scale,
            data,
        )


class DummyGraphCV(nn.Module):
    def __init__(
        self,
        pooling_operation: str | None = "mean",
    ) -> None:
        super().__init__()
        self.in_features = None
        self.nn = DummyGraphEncoder(pooling_operation)
        self.preprocessing = GraphShiftPreprocessing()
        self.norm_in = None


def _vector_input() -> torch.Tensor:
    return torch.tensor(
        [
            [1.0, 2.0, 3.0],
            [2.0, 0.0, 1.0],
        ]
    )


def _graph_input(
    dtype: torch.dtype = torch.float64,
) -> dict[str, torch.Tensor]:
    return {
        "positions": torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [2.0, 1.0, 0.0],
                [1.0, 2.0, 0.0],
            ],
            dtype=dtype,
        ),
        "batch": torch.tensor([0, 0, 1]),
        "ptr": torch.tensor([0, 2, 3]),
        "edge_index": torch.tensor([[0, 1], [1, 0]]),
        "unit_shifts": torch.zeros(2, 3, dtype=dtype),
    }


def _vector_representation(
    freeze: bool = True,
) -> MLColvarRepresentation:
    return MLColvarRepresentation(
        DummyDescriptorCV(),
        freeze=freeze,
    )


def _graph_representation(
    pooling_operation: str | None = "mean",
) -> MLColvarRepresentation:
    return MLColvarRepresentation(
        DummyGraphCV(pooling_operation),
    )


def _regression_model(
    representation: Representation,
) -> RegressionCV:
    return RegressionCV(
        model=FeedForward([representation.out_features, 1]),
        preprocessing=representation,
    )


def _postprocessing(
    representation: Representation,
) -> nn.Module:
    return nn.Sequential(
        FeedForward([representation.out_features, 1]),
        nn.Sigmoid(),
    ).eval()


def test_vector_representation() -> None:
    representation = _vector_representation()
    output = representation(
        _vector_input(),
        cell=torch.tensor(0.5),
    )

    torch.testing.assert_close(
        output,
        torch.tensor(
            [
                [5.0, 16.0],
                [7.0, 8.0],
            ]
        ),
    )
    assert representation.input_kind == "vector"
    assert representation.output_kind == "system"
    assert representation.in_features == 3
    assert representation.out_features == 2


def test_vector_freeze_and_gradients() -> None:
    pretrained = DummyDescriptorCV()
    representation = MLColvarRepresentation(pretrained)

    x = _vector_input().requires_grad_(True)
    representation(x).sum().backward()

    assert x.grad is not None
    assert all(
        not parameter.requires_grad
        for parameter in representation.parameters()
    )
    assert all(
        parameter.grad is None
        for parameter in representation.parameters()
    )

    representation.train()
    assert not representation.training
    assert not representation.encoder.training

    unfrozen = _vector_representation(freeze=False)
    assert all(
        parameter.requires_grad
        for parameter in unfrozen.parameters()
    )


def test_vector_representation_preprocessing() -> None:
    representation = _vector_representation()
    model = _regression_model(representation)

    x = _vector_input().requires_grad_(True)
    output = model(x)
    output.sum().backward()

    assert output.shape == (2, 1)
    assert x.grad is not None
    assert all(
        parameter.grad is None
        for parameter in representation.parameters()
    )
    assert any(
        parameter.grad is not None
        for parameter in model.nn.parameters()
    )


def test_graph_representation() -> None:
    representation = _graph_representation()
    output = representation(_graph_input())

    torch.testing.assert_close(
        output,
        torch.tensor(
            [
                [4.0, 3.0],
                [4.0, 6.0],
            ],
            dtype=torch.float32,
        ),
    )
    assert representation.input_kind == "graph"
    assert representation.output_kind == "system"
    assert representation.out_features == 2
    assert representation.pooling_operation == "mean"
    assert representation.atomic_numbers.tolist() == [1, 6, 8]
    assert representation.cutoff.item() == 4.0
    assert representation.buffer.item() == 0.5


def test_graph_atom_representation() -> None:
    representation = _graph_representation(
        pooling_operation=None
    )

    torch.testing.assert_close(
        representation(_graph_input()),
        torch.tensor(
            [
                [2.0, 2.0],
                [6.0, 4.0],
                [4.0, 6.0],
            ],
            dtype=torch.float32,
        ),
    )
    assert representation.output_kind == "atom"
    assert representation.pooling_operation is None


def test_graph_representation_gradients() -> None:
    representation = _graph_representation()
    model = _regression_model(representation)

    data = _graph_input()
    data["positions"].requires_grad_(True)
    model(data).sum().backward()

    assert data["positions"].grad is not None
    assert all(
        parameter.grad is None
        for parameter in representation.parameters()
    )
    assert any(
        parameter.grad is not None
        for parameter in model.nn.parameters()
    )


def test_vector_representation_torchscript(tmp_path) -> None:
    representation = _vector_representation()
    path = tmp_path / "vector_model.ptc"

    export_representation_torchscript(
        representation,
        path,
        postprocessing=_postprocessing(representation),
    )

    loaded = torch.jit.load(str(path), map_location="cpu").eval()
    assert path.exists()
    assert loaded(_vector_input()).shape == (2, 1)


def test_graph_representation_torchscript(tmp_path) -> None:
    representation = _graph_representation()
    graph = _graph_input()
    graph["node_attrs"] = torch.ones(
        graph["positions"].shape[0],
        1,
        dtype=graph["positions"].dtype,
    )
    graph["shifts"] = graph["unit_shifts"].clone()
    path = tmp_path / "graph_model.ptc"

    export_representation_torchscript(
        representation,
        path,
        postprocessing=_postprocessing(representation),
        example_input=graph,
    )

    loaded = torch.jit.load(str(path), map_location="cpu").eval()
    assert path.exists()
    assert loaded(graph).shape == (2, 1)