import pytest
import torch
from torch import nn
from torch_geometric.data import Data

from mlcolvar.data import DictDataset
from mlcolvar.representation import Representation, evaluate_dataset, prepare_committor


class VectorRepresentation(Representation):
    def __init__(self):
        super().__init__(
            in_features=2,
            out_features=2,
            input_kind="vector",
            output_kind="system",
        )

    def forward(self, x, cell=None):
        return x


class GraphRepresentation(Representation):
    def __init__(self):
        super().__init__(
            out_features=2,
            input_kind="graph",
            output_kind="system",
            atomic_numbers=[1],
            cutoff=5.0,
            pooling_operation="mean",
        )

    def forward(self, data, cell=None):
        return self.pooling(data["positions"][:, :2], data)


class DescriptorDerivatives(nn.Module):
    def forward(self, gradient, ref_idx=None):
        if ref_idx is None:
            raise ValueError("`ref_idx` is required.")
        output = torch.zeros(gradient.shape[0], 1, 3, gradient.shape[-1])
        output[:, 0, :2] = gradient
        return output


@pytest.mark.parametrize(
    ("derivatives", "expected"),
    [
        (None, torch.ones(2, 2)),
        (
            DescriptorDerivatives(),
            torch.tensor([[[1., 1., 0.]], [[1., 1., 0.]]]),
        ),
    ],
)
def test_vector_committor_preparation(derivatives, expected):
    data = {
        "data": torch.tensor([[0., 1.], [1., 2.], [2., 3.], [3., 4.]]),
        "labels": torch.tensor([0., 2., 1., 3.]),
        "weights": torch.ones(4),
    }
    if derivatives is not None:
        data["ref_idx"] = torch.arange(4)

    dataset = DictDataset(data)
    representation = VectorRepresentation()
    features = evaluate_dataset(representation, dataset)
    prepared, transform = prepare_committor(
        representation,
        dataset,
        features,
        descriptor_derivatives=derivatives,
    )

    torch.testing.assert_close(prepared["data"], features)
    torch.testing.assert_close(
        prepared["ref_idx"],
        torch.tensor([-1, 0, -1, 1], dtype=prepared["ref_idx"].dtype),
    )
    torch.testing.assert_close(
        transform(torch.ones(2, 2), torch.tensor([0, 1])),
        expected,
    )


def test_graph_committor_preparation():
    dataset = DictDataset(
        {
            "data_list": [
                Data(
                    positions=torch.tensor([[1., 0., 0.], [3., 2., 0.]]),
                    graph_labels=torch.tensor([2.]),
                    weight=torch.tensor([1.]),
                    num_nodes=2,
                ),
                Data(
                    positions=torch.tensor([[2., 1., 0.], [4., 3., 0.]]),
                    graph_labels=torch.tensor([3.]),
                    weight=torch.tensor([2.]),
                    num_nodes=2,
                ),
            ]
        },
        data_type="graphs",
    )

    representation = GraphRepresentation()
    features = evaluate_dataset(representation, dataset)
    prepared, transform = prepare_committor(representation, dataset, features)

    torch.testing.assert_close(
        features,
        torch.tensor([[2., 1.], [3., 2.]]),
    )
    torch.testing.assert_close(
        prepared["weights"],
        torch.tensor([1., 2.]),
    )

    expected = torch.full((2, 2, 3), 0.5)
    expected[:, :, 2] = 0.0
    torch.testing.assert_close(
        transform(torch.ones(2, 2), torch.tensor([0, 1])),
        expected,
    )