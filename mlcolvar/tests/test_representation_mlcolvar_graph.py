from typing import Dict, Optional

import pytest
import torch
from torch import nn

from mlcolvar.core import BaseGNN
from mlcolvar.representation import (
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

        output["positions"] = (
            data["positions"] + shift
        )
        return output


class DummyGraphEncoder(BaseGNN):
    def __init__(
        self,
        pooling_operation: Optional[str] = "mean",
        dtype: torch.dtype = torch.float32,
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

        self.scale = nn.Parameter(
            torch.tensor(2.0, dtype=dtype)
        )
        self.to(dtype=dtype)

    def forward(
        self,
        data: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        positions = data["positions"]
        batch = data["batch"]
        ptr = data["ptr"]

        node_features = (
            positions[:, :2] * self.scale
        )

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
            ).to(
                dtype=output.dtype,
                device=output.device,
            )
            output = (
                output
                / counts.clamp_min(1.0)
                .unsqueeze(-1)
            )

        elif self.pooling_operation != "sum":
            raise ValueError(
                "Unsupported dummy pooling operation."
            )

        return output


class DummyGraphCV(nn.Module):
    def __init__(
        self,
        pooling_operation: Optional[str] = "mean",
        use_preprocessing: bool = True,
        use_norm_in: bool = False,
    ) -> None:
        super().__init__()

        self.in_features = None
        self.nn = DummyGraphEncoder(
            pooling_operation=pooling_operation
        )

        if use_preprocessing:
            self.preprocessing = (
                GraphShiftPreprocessing()
            )

        self.norm_in = (
            nn.Identity()
            if use_norm_in
            else None
        )


class DummyVectorCV(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.in_features = 3
        self.out_features = 2
        self.nn = nn.Linear(3, 2)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return self.nn(x)


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
    **kwargs,
) -> MLColvarRepresentation:
    return MLColvarRepresentation(
        DummyGraphCV(**kwargs),
        mode="latent",
        freeze=True,
    )


def test_graph_representation() -> None:
    representation = make_representation()

    data = make_graph()
    output = representation(data)

    expected = torch.tensor(
        [
            [4.0, 3.0],
            [4.0, 6.0],
        ],
        dtype=torch.float32,
    )

    torch.testing.assert_close(
        output,
        expected,
    )

    assert representation.input_kind == "graph"
    assert representation.output_kind == "system"
    assert representation.out_features == 2
    assert representation.pooling_operation == "mean"
    assert representation.atomic_numbers.tolist() == [
        1,
        6,
        8,
    ]

    torch.testing.assert_close(
        representation.cutoff,
        torch.tensor(4.0),
    )

    state = representation.state_dict()

    assert "model.nn.scale" in state
    assert "cutoff" in state
    assert "atomic_numbers" in state
    assert "_model_reference" not in state

    # Input is not modified in place.
    assert data["positions"].dtype == torch.float64


def test_graph_representation_preserves_gradients() -> None:
    pretrained = DummyGraphCV()

    representation = MLColvarRepresentation(
        pretrained,
        mode="latent",
        freeze=True,
    )

    data = make_graph()
    data["positions"].requires_grad_(True)

    representation(
        data
    ).sum().backward()

    assert data["positions"].grad is not None
    assert torch.isfinite(
        data["positions"].grad
    ).all()

    assert all(
        not p.requires_grad
        for p in pretrained.parameters()
    )
    assert all(
        p.grad is None
        for p in pretrained.parameters()
    )

    representation.train()

    assert not representation.training
    assert not pretrained.training
    assert not pretrained.nn.training


def test_representation_model_trains_only_head() -> None:
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

    assert isinstance(model, BaseGNN)

    data = make_graph()
    data["positions"].requires_grad_(True)

    output = model(data)

    assert output.shape == (2, 1)

    output.sum().backward()

    assert data["positions"].grad is not None
    assert pretrained.nn.scale.grad is None

    assert any(
        p.grad is not None
        for p in model.head.parameters()
    )


def test_graph_requires_pooling() -> None:
    with pytest.raises(
        ValueError,
        match="graph-level encoder",
    ):
        make_representation(
            pooling_operation=None
        )


def test_graph_rejects_norm_in() -> None:
    representation = make_representation(
        use_norm_in=True
    )

    with pytest.raises(
        ValueError,
        match="Input normalization",
    ):
        representation(
            make_graph()
        )


def test_factory_distinguishes_vector_and_graph() -> None:
    vector_representation = (
        MLColvarRepresentation(
            DummyVectorCV(),
            mode="latent",
        )
    )

    graph_representation = (
        make_representation()
    )

    assert (
        vector_representation.input_kind
        == "vector"
    )
    assert (
        graph_representation.input_kind
        == "graph"
    )

    with pytest.raises(
        ValueError,
        match="mode='latent'",
    ):
        MLColvarRepresentation(
            DummyGraphCV(),
            mode="output",
        )


def test_graph_representation_torchscript_roundtrip(
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

    postprocessing = nn.Sigmoid()
    path = tmp_path / "graph_representation.ptc"

    graph = make_graph()

    graph["node_attrs"] = torch.ones(
        graph["positions"].shape[0],
        1,
        dtype=graph["positions"].dtype,
    )
    graph["shifts"] = (
        graph["unit_shifts"].clone()
    )

    export_representation_torchscript(
        model=model,
        postprocessing=postprocessing,
        path=path,
        example_input=graph,
    )

    loaded = torch.jit.load(
        str(path),
        map_location="cpu",
    ).eval()

    with torch.no_grad():
        expected = postprocessing(
            model(graph)
        )
        output = loaded(graph)

    assert path.exists()

    torch.testing.assert_close(
        output,
        expected,
        rtol=1e-5,
        atol=1e-6,
    )