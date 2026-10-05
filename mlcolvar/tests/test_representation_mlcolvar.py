import pytest
import torch
from torch import nn

from mlcolvar.core import BaseGNN, FeedForward
from mlcolvar.cvs import RegressionCV
from mlcolvar.representation import MLColvarRepresentation


class AddOne(nn.Module):
    def forward(self, x, cell=None):
        return x + 1 if cell is None else x + 1 + cell.reshape(())


class ScaleTwo(nn.Module):
    def forward(self, x):
        return 2 * x


class DummyVectorCV(nn.Module):
    def __init__(self):
        super().__init__()
        self.in_features = 3
        self.preprocessing = AddOne()
        self.norm_in = ScaleTwo()
        self.nn = nn.Linear(3, 2, bias=False)
        with torch.no_grad():
            self.nn.weight.copy_(torch.tensor([[1., 0., 0.], [0., 1., 1.]]))


class GraphShift(nn.Module):
    def forward(self, data, cell=None):
        data = dict(data)
        data["positions"] = data["positions"] + (
            1 if cell is None else 1 + cell.reshape(())
        )
        return data


class DummyGraphEncoder(BaseGNN):
    def __init__(self, pooling_operation="mean"):
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

    def forward(self, data):
        return self.pooling(data["positions"][:, :2] * self.scale, data)


class DummyGraphCV(nn.Module):
    def __init__(self, pooling_operation="mean"):
        super().__init__()
        self.in_features = None
        self.nn = DummyGraphEncoder(pooling_operation)
        self.preprocessing = GraphShift()
        self.norm_in = None


def vector_input():
    return torch.tensor([[1., 2., 3.], [2., 0., 1.]])


def graph_input():
    return {
        "positions": torch.tensor([[0., 0., 0.], [2., 1., 0.], [1., 2., 0.]]),
        "batch": torch.tensor([0, 0, 1]),
        "ptr": torch.tensor([0, 2, 3]),
        "edge_index": torch.tensor([[0, 1], [1, 0]]),
        "unit_shifts": torch.zeros(2, 3),
    }


def vector_representation(freeze=True):
    return MLColvarRepresentation(DummyVectorCV(), freeze=freeze)


def graph_representation(pooling_operation="mean"):
    return MLColvarRepresentation(DummyGraphCV(pooling_operation))


def regression_model(representation):
    return RegressionCV(
        model=FeedForward([representation.out_features, 1]),
        preprocessing=representation,
    )


def test_vector_representation():
    representation = vector_representation()
    x = vector_input().requires_grad_(True)
    output = representation(x, cell=torch.tensor(0.5))

    torch.testing.assert_close(
        output, torch.tensor([[5., 16.], [7., 8.]])
    )
    output.sum().backward()

    assert x.grad is not None
    assert representation.input_kind == "vector"
    assert representation.output_kind == "system"
    assert representation.in_features == 3
    assert representation.out_features == 2
    assert all(
        not p.requires_grad for p in representation.parameters()
    )
    assert all(
        p.requires_grad
        for p in vector_representation(freeze=False).parameters()
    )


@pytest.mark.parametrize(
    ("pooling", "output_kind", "expected"),
    [
        (
            "mean",
            "system",
            torch.tensor([[4., 3.], [4., 6.]]),
        ),
        (
            None,
            "atom",
            torch.tensor([[2., 2.], [6., 4.], [4., 6.]]),
        ),
    ],
)
def test_graph_representation(pooling, output_kind, expected):
    representation = graph_representation(pooling)
    output = representation(graph_input())

    torch.testing.assert_close(output, expected)

    assert representation.input_kind == "graph"
    assert representation.output_kind == output_kind
    assert representation.pooling_operation == pooling
    assert representation.out_features == 2


def test_representation_as_preprocessing():
    representation = graph_representation()
    model = regression_model(representation)
    data = graph_input()
    data["positions"].requires_grad_(True)

    output = model(data)
    output.sum().backward()

    assert output.shape == (2, 1)
    assert data["positions"].grad is not None
    assert all(p.grad is None for p in representation.parameters())
    assert any(p.grad is not None for p in model.nn.parameters())


def test_graph_representation_torchscript(tmp_path):
    model = regression_model(graph_representation()).eval()
    graph = graph_input()
    graph["node_attrs"] = torch.ones(len(graph["positions"]), 1)
    graph["shifts"] = graph["unit_shifts"].clone()

    path = tmp_path / "model.ptc"
    model.to_torchscript(
        file_path=path,
        method="trace",
        example_inputs=graph,
    )
    loaded = torch.jit.load(str(path)).eval()

    assert path.exists()
    assert loaded(graph).shape == (2, 1)