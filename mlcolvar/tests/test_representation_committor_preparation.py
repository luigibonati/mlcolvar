import torch
from torch import nn
from torch_geometric.data import Data

from mlcolvar.data import DictDataset
from mlcolvar.representation import (
    Representation,
    evaluate_dataset,
    prepare_committor,
)


class DummyVectorRepresentation(Representation):
    def __init__(self):
        super().__init__(
            in_features=2,
            out_features=2,
            input_kind="vector",
        )

    def forward(self, x, cell=None):
        return x


class DummyGraphRepresentation(Representation):
    def __init__(self):
        super().__init__(
            out_features=2,
            input_kind="graph",
            atomic_numbers=[1],
            cutoff=5.0,
            pooling_operation="mean",
        )

    def forward(self, data, cell=None):
        return self.pooling(data["positions"][:, :2], data)


class DummyDescriptorDerivatives(nn.Module):
    def forward(self, gradient, ref_idx=None):
        if ref_idx is None:
            raise ValueError("`ref_idx` is required.")

        output = torch.zeros(
            gradient.shape[0],
            1,
            3,
            gradient.shape[-1],
            device=gradient.device,
            dtype=gradient.dtype,
        )
        output[:, 0, 0, :] = gradient[:, 0, :]
        output[:, 0, 1, :] = gradient[:, 1, :]
        return output


def _vector_dataset(ref_idx: bool = False) -> DictDataset:
    data = {
        "data": torch.tensor(
            [
                [0.0, 1.0],
                [1.0, 2.0],
                [2.0, 3.0],
                [3.0, 4.0],
            ]
        ),
        "labels": torch.tensor([0.0, 2.0, 1.0, 3.0]),
        "weights": torch.ones(4),
    }
    if ref_idx:
        data["ref_idx"] = torch.arange(4)
    return DictDataset(data)


def test_vector_committor_preparation():
    dataset = _vector_dataset()
    representation = DummyVectorRepresentation()
    features = evaluate_dataset(representation, dataset)
    prepared, transform = prepare_committor(
        representation,
        dataset,
        features,
    )

    torch.testing.assert_close(prepared["data"], features)
    torch.testing.assert_close(
        prepared["ref_idx"],
        torch.tensor(
            [-1, 0, -1, 1],
            dtype=prepared["ref_idx"].dtype,
        ),
    )

    gradient = transform(
        torch.ones(2, 2),
        torch.tensor([0, 1]),
    )
    torch.testing.assert_close(gradient, torch.ones(2, 2))


def test_vector_descriptor_committor_preparation():
    dataset = _vector_dataset(ref_idx=True)
    representation = DummyVectorRepresentation()
    features = evaluate_dataset(representation, dataset)
    prepared, transform = prepare_committor(
        representation,
        dataset,
        features,
        descriptor_derivatives=DummyDescriptorDerivatives(),
    )

    torch.testing.assert_close(prepared["data"], features)
    torch.testing.assert_close(
        prepared["ref_idx"],
        torch.tensor(
            [-1, 0, -1, 1],
            dtype=prepared["ref_idx"].dtype,
        ),
    )

    gradient = transform(
        torch.ones(2, 2),
        torch.tensor([0, 1]),
    )
    expected = torch.tensor(
        [
            [[1.0, 1.0, 0.0]],
            [[1.0, 1.0, 0.0]],
        ]
    )
    torch.testing.assert_close(gradient, expected)


def test_graph_committor_preparation():
    graphs = [
        Data(
            positions=torch.tensor(
                [
                    [1.0, 0.0, 0.0],
                    [3.0, 2.0, 0.0],
                ]
            ),
            graph_labels=torch.tensor([2.0]),
            weight=torch.tensor([1.0]),
            num_nodes=2,
        ),
        Data(
            positions=torch.tensor(
                [
                    [2.0, 1.0, 0.0],
                    [4.0, 3.0, 0.0],
                ]
            ),
            graph_labels=torch.tensor([3.0]),
            weight=torch.tensor([2.0]),
            num_nodes=2,
        ),
    ]
    dataset = DictDataset(
        {"data_list": graphs},
        data_type="graphs",
    )

    representation = DummyGraphRepresentation()
    features = evaluate_dataset(representation, dataset)
    prepared, transform = prepare_committor(
        representation,
        dataset,
        features,
    )

    torch.testing.assert_close(
        features,
        torch.tensor(
            [
                [2.0, 1.0],
                [3.0, 2.0],
            ]
        ),
    )
    torch.testing.assert_close(prepared["data"], features)
    torch.testing.assert_close(
        prepared["weights"],
        torch.tensor([1.0, 2.0]),
    )
    torch.testing.assert_close(
        prepared["ref_idx"],
        torch.tensor(
            [0, 1],
            dtype=prepared["ref_idx"].dtype,
        ),
    )

    gradient = transform(
        torch.ones(2, 2),
        torch.tensor([0, 1]),
    )
    expected = torch.tensor(
        [
            [
                [0.5, 0.5, 0.0],
                [0.5, 0.5, 0.0],
            ],
            [
                [0.5, 0.5, 0.0],
                [0.5, 0.5, 0.0],
            ],
        ]
    )
    torch.testing.assert_close(gradient, expected)