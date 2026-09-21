from typing import Dict, Optional

import torch
from torch import nn

from mlcolvar.core import BaseGNN
from mlcolvar.representation import (
    GraphRepresentation,
    MLColvarRepresentation,
    RepresentationModel,
    TaskHead,
    export_representation_torchscript,
)


class GraphShiftPreprocessing(nn.Module):
    def forward(
        self,
        data: Dict[str, torch.Tensor],
        cell: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        output = dict(data)

        shift = 1.0
        if cell is not None:
            shift += float(cell.reshape(()).item())

        output["positions"] = data["positions"] + shift
        return output


class DummyGraphEncoder(BaseGNN):
    def __init__(
        self,
        pooling_operation: Optional[str] = "mean",
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
        data: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        positions = data["positions"]
        batch = data["batch"]
        ptr = data["ptr"]

        node_features = positions[:, :2] * self.scale

        if self.pooling_operation is None:
            return node_features

        n_graphs = ptr.numel() - 1
        output = node_features.new_zeros(
            (n_graphs, 2)
        )

        output.index_add_(
            0,
            batch,
            node_features,
        )

        if self.pooling_operation == "mean":
            counts = torch.bincount(
                batch,
                minlength=n_graphs,
            ).to(output)

            output = (
                output
                / counts.clamp_min(1).unsqueeze(-1)
            )

        return output


class DummyGraphCV(nn.Module):
    def __init__(
        self,
        pooling_operation: Optional[str] = "mean",
    ) -> None:
        super().__init__()

        self.in_features = None
        self.nn = DummyGraphEncoder(
            pooling_operation=pooling_operation
        )
        self.preprocessing = GraphShiftPreprocessing()
        self.norm_in = None


def make_graph(
    dtype: torch.dtype = torch.float64,
) -> Dict[str, torch.Tensor]:
    return {
        "positions": torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [2.0, 1.0, 0.0],
                [1.0, 2.0, 0.0],
            ],
            dtype=dtype,
        ),
        "batch": torch.tensor(
            [0, 0, 1],
            dtype=torch.long,
        ),
        "ptr": torch.tensor(
            [0, 2, 3],
            dtype=torch.long,
        ),
        "edge_index": torch.tensor(
            [
                [0, 1],
                [1, 0],
            ],
            dtype=torch.long,
        ),
        "unit_shifts": torch.zeros(
            2,
            3,
            dtype=dtype,
        ),
    }


def make_representation(
    pooling_operation: Optional[str] = "mean",
) -> GraphRepresentation:
    return MLColvarRepresentation(
        DummyGraphCV(
            pooling_operation=pooling_operation
        ),
        mode="latent",
        freeze=True,
    )


def test_graph_representation() -> None:
    representation = make_representation()

    output = representation(
        make_graph()
    )

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

    assert representation.output_kind == "system"
    assert representation.out_features == 2
    assert representation.pooling_operation == "mean"


def test_graph_atom_representation() -> None:
    representation = make_representation(
        pooling_operation=None
    )

    output = representation(
        make_graph()
    )

    torch.testing.assert_close(
        output,
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

    pooled = representation.pool("mean")

    torch.testing.assert_close(
        pooled(make_graph()),
        torch.tensor(
            [
                [4.0, 3.0],
                [4.0, 6.0],
            ],
            dtype=torch.float32,
        ),
    )

    concatenated = representation.concat_atoms([0])

    torch.testing.assert_close(
        concatenated(make_graph()),
        torch.tensor(
            [
                [2.0, 2.0],
                [4.0, 6.0],
            ],
            dtype=torch.float32,
        ),
    )

    assert pooled.output_kind == "system"
    assert concatenated.output_kind == "system"


def test_graph_representation_gradients() -> None:
    pretrained = DummyGraphCV()

    representation = MLColvarRepresentation(
        pretrained,
        mode="latent",
        freeze=True,
    )

    model = RepresentationModel(
        representation,
        n_out=1,
        hidden_layers=(),
    )

    data = make_graph()
    data["positions"].requires_grad_(True)

    model(data).sum().backward()

    assert data["positions"].grad is not None

    assert all(
        parameter.grad is None
        for parameter in pretrained.parameters()
    )

    assert any(
        parameter.grad is not None
        for parameter in model.head.parameters()
    )


def test_graph_representation_torchscript(
    tmp_path,
) -> None:
    representation = make_representation()

    model = RepresentationModel(
        representation,
        head=TaskHead(
            representation.out_features,
            n_out=1,
            hidden_layers=(),
        ),
    ).eval()

    graph = make_graph()

    graph["node_attrs"] = torch.ones(
        graph["positions"].shape[0],
        1,
        dtype=graph["positions"].dtype,
    )
    graph["shifts"] = graph["unit_shifts"].clone()

    path = tmp_path / "model.ptc"

    export_representation_torchscript(
        model=model,
        postprocessing=nn.Sigmoid(),
        path=path,
        example_input=graph,
    )

    loaded = torch.jit.load(
        str(path),
        map_location="cpu",
    ).eval()

    assert path.exists()
    assert loaded(graph).shape == (2, 1)