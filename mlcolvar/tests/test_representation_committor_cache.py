import torch
from torch_geometric.data import Data

from mlcolvar.data import DictDataset
from mlcolvar.representation import (
    GraphRepresentation,
    VectorRepresentation,
)
from mlcolvar.representation.cache import IdentityDescriptorDerivatives
from mlcolvar.representation.committor_cache import precompute_committor_cache


class DummyVectorRepresentation(VectorRepresentation):
    def __init__(self):
        super().__init__(
            in_features=2,
            out_features=2,
            freeze=True,
        )

    def forward(self, x, cell=None):
        return x


class DummyGraphRepresentation(GraphRepresentation):
    def __init__(self):
        super().__init__(
            out_features=2,
            atomic_numbers=[1],
            cutoff=5.0,
            pooling_operation="mean",
            freeze=True,
        )

    def forward(self, data, cell=None):
        return self.pooling(
            data["positions"][:, :2],
            data,
        )


def test_vector_committor_cache():
    dataset = DictDataset(
        {
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
            "ref_idx": torch.arange(4),
        }
    )

    cached, derivatives = precompute_committor_cache(
        DummyVectorRepresentation(),
        dataset,
        descriptor_derivatives=IdentityDescriptorDerivatives(),
    )

    torch.testing.assert_close(
        cached["data"],
        dataset["data"],
    )
    torch.testing.assert_close(
        cached["ref_idx"],
        torch.tensor([-1, 0, -1, 1], dtype=torch.int),
    )

    gradient = derivatives(
        torch.ones(2, 2),
        torch.tensor([0, 1]),
    )

    assert gradient.shape == (2, 1, 2)


def test_graph_committor_cache():
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

    cached, derivatives = precompute_committor_cache(
        DummyGraphRepresentation(),
        dataset,
    )

    torch.testing.assert_close(
        cached["data"],
        torch.tensor(
            [
                [2.0, 1.0],
                [3.0, 2.0],
            ]
        ),
    )
    torch.testing.assert_close(
        cached["weights"],
        torch.tensor([1.0, 2.0]),
    )
    torch.testing.assert_close(
        cached["ref_idx"],
        torch.tensor([0, 1], dtype=torch.int),
    )

    gradient = derivatives(
        torch.ones(2, 2),
        torch.tensor([0, 1]),
    )

    assert gradient.shape == (2, 2, 3)