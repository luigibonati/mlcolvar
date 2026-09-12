from __future__ import annotations

from typing import Dict, Optional

import pytest
import torch
from torch import nn
from torch_geometric.data import Data

from mlcolvar.core import BaseGNN
from mlcolvar.data import DictDataset
from mlcolvar.representation import (
    MLColvarRepresentation,
    RepresentationModel,
    TaskHead,
    export_representation_torchscript,
    precompute_committor_cache,
)


class GraphShiftPreprocessing(nn.Module):
    def forward(
        self,
        data: Dict[str, torch.Tensor],
        cell: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        output = {
            key: value
            for key, value in data.items()
        }

        shift = 1.0
        if cell is not None:
            shift = shift + float(
                cell.reshape(()).item()
            )

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
            torch.tensor(
                2.0,
                dtype=dtype,
            )
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
            positions[:, :2]
            * self.scale
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
            output = output / counts.clamp_min(
                1.0
            ).unsqueeze(-1)

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
            pooling_operation=pooling_operation,
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


class DummyTensorCV(nn.Module):
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
            (2, 3),
            dtype=dtype,
        ),
    }


def test_graph_mlcolvar_representation_features_metadata_and_dtype() -> None:
    pretrained = DummyGraphCV()

    representation = MLColvarRepresentation(
        pretrained,
        mode="latent",
        freeze=True,
    )

    data = make_graph()
    output = representation(data)

    expected = torch.tensor(
        [
            [4.0, 3.0],
            [4.0, 6.0],
        ],
        dtype=torch.float32,
    )

    assert representation.input_kind == "graph"
    assert representation.output_kind == "system"
    assert representation.in_features is None
    assert representation.out_features == 2
    assert representation.pooling_operation == "mean"

    torch.testing.assert_close(
        representation.cutoff,
        torch.tensor(4.0),
    )
    torch.testing.assert_close(
        representation.buffer,
        torch.tensor(0.5),
    )
    torch.testing.assert_close(
        representation.long_range_cutoff,
        torch.tensor(-1.0),
    )

    assert representation.atomic_numbers.tolist() == [
        1,
        6,
        8,
    ]

    assert output.shape == (2, 2)
    assert output.dtype == torch.float32
    torch.testing.assert_close(
        output,
        expected,
    )

    # Input is not modified in-place.
    assert data["positions"].dtype == torch.float64
    torch.testing.assert_close(
        data["positions"],
        make_graph()["positions"],
    )


def test_frozen_graph_representation_preserves_position_gradients() -> None:
    pretrained = DummyGraphCV()

    representation = MLColvarRepresentation(
        pretrained,
        mode="latent",
        freeze=True,
    )

    data = make_graph()
    data["positions"].requires_grad_(True)

    output = representation(data)
    output.sum().backward()

    assert data["positions"].grad is not None
    assert torch.isfinite(
        data["positions"].grad
    ).all()
    assert torch.count_nonzero(
        data["positions"].grad
    ) > 0

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


def test_graph_representation_state_dict_contains_model_and_metadata() -> None:
    representation = MLColvarRepresentation(
        DummyGraphCV(),
        mode="latent",
        freeze=True,
    )

    state = representation.state_dict()

    assert "model.nn.scale" in state
    assert "cutoff" in state
    assert "buffer" in state
    assert "long_range_cutoff" in state
    assert "atomic_numbers" in state
    assert "_model_reference" not in state


def test_graph_representation_model_is_basegnn_and_trains_only_head() -> None:
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

    assert isinstance(
        model,
        BaseGNN,
    )

    assert model.representation is representation
    assert model.out_features == 1
    assert "_radial_embedding" not in model._modules

    torch.testing.assert_close(
        model.cutoff,
        representation.cutoff,
    )
    torch.testing.assert_close(
        model.buffer,
        representation.buffer,
    )
    torch.testing.assert_close(
        model.long_range_cutoff,
        representation.long_range_cutoff,
    )

    trainable = [
        p
        for p in model.parameters()
        if p.requires_grad
    ]
    assert sum(
        p.numel()
        for p in trainable
    ) == 3

    linear = next(
        module
        for module in model.head.modules()
        if isinstance(module, nn.Linear)
    )

    with torch.no_grad():
        linear.weight.fill_(1.0)
        linear.bias.zero_()

    data = make_graph()
    data["positions"].requires_grad_(True)

    output = model(data)

    assert output.shape == (2, 1)
    output.sum().backward()

    assert data["positions"].grad is not None
    assert torch.isfinite(
        data["positions"].grad
    ).all()
    assert linear.weight.grad is not None
    assert pretrained.nn.scale.grad is None


def test_graph_representation_model_can_be_traced() -> None:
    model = RepresentationModel(
        MLColvarRepresentation(
            DummyGraphCV(),
            mode="latent",
            freeze=True,
        ),
        n_out=1,
        hidden_layers=(),
    ).eval()

    data = make_graph()
    expected = model(data)

    traced = torch.jit.trace(
        model,
        example_inputs=(data,),
        strict=False,
        check_trace=True,
    )

    output = traced(data)

    torch.testing.assert_close(
        output,
        expected,
    )


def test_graph_representation_requires_graph_level_pooling() -> None:
    with pytest.raises(
        ValueError,
        match="graph-level encoder",
    ):
        MLColvarRepresentation(
            DummyGraphCV(
                pooling_operation=None,
            ),
            mode="latent",
        )


def test_graph_representation_rejects_tensor_norm_in() -> None:
    representation = MLColvarRepresentation(
        DummyGraphCV(
            use_norm_in=True,
        ),
        mode="latent",
    )

    with pytest.raises(
        ValueError,
        match="Tensor input normalization",
    ):
        representation(
            make_graph()
        )


def test_mlcolvar_representation_factory_distinguishes_tensor_and_graph() -> None:
    tensor_rep = MLColvarRepresentation(
        DummyTensorCV(),
        mode="latent",
    )
    graph_rep = MLColvarRepresentation(
        DummyGraphCV(),
        mode="latent",
    )

    assert tensor_rep.input_kind == "tensor"
    assert graph_rep.input_kind == "graph"

    with pytest.raises(
        ValueError,
        match="support only mode='latent'",
    ):
        MLColvarRepresentation(
            DummyGraphCV(),
            mode="output",
        )


def make_graph_sample(
    offset: float,
    label: float = 2.0,
) -> Data:
    positions = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [2.0, 1.0, 0.0],
            [1.0, 2.0, 0.0],
        ],
        dtype=torch.float32,
    ) + offset

    return Data(
        positions=positions,
        num_nodes=positions.shape[0],
        edge_index=torch.tensor(
            [
                [0, 1],
                [1, 0],
            ],
            dtype=torch.long,
        ),
        unit_shifts=torch.zeros(
            (2, 3),
            dtype=torch.float32,
        ),
        graph_labels=torch.tensor(
            [label],
            dtype=torch.float32,
        ),
        weight=torch.ones(
            1,
            dtype=torch.float32,
        ),
    )


def test_graph_committor_cache_matches_direct_representation() -> None:
    dataset = DictDataset(
        {
            "data_list": [
                make_graph_sample(0.0),
                make_graph_sample(0.5),
            ]
        },
        metadata={
            "atomic_numbers": [1, 6, 8],
        },
        data_type="graphs",
    )

    representation = MLColvarRepresentation(
        DummyGraphCV(
            use_preprocessing=False,
        ),
        mode="latent",
        freeze=True,
    )

    cached_dataset, cached_derivatives = (
        precompute_committor_cache(
            representation=representation,
            dataset=dataset,
            batch_size=1,
            device="cpu",
            output_device="cpu",
            separate_boundary_dataset=False,
        )
    )

    graph_batch = dataset.get_graph_inputs()

    with torch.no_grad():
        expected_latent = representation(
            graph_batch
        )

    torch.testing.assert_close(
        cached_dataset["data"],
        expected_latent,
    )

    assert cached_dataset["data"].shape == (2, 2)
    assert cached_derivatives.jacobian.shape == (
        2,
        3,
        3,
        2,
    )

    expected_jacobian = torch.zeros(
        2,
        3,
        3,
        2,
    )

    expected_jacobian[:, :, 0, 0] = 2.0 / 3.0
    expected_jacobian[:, :, 1, 1] = 2.0 / 3.0

    torch.testing.assert_close(
        cached_derivatives.jacobian,
        expected_jacobian,
    )


def test_graph_representation_torchscript_roundtrip(
    tmp_path,
) -> None:
    representation = MLColvarRepresentation(
        DummyGraphCV(),
        mode="latent",
        freeze=True,
    )

    head = TaskHead(
        representation.out_features,
        n_out=1,
        hidden_layers=(),
    )

    model = RepresentationModel(
        representation,
        head=head,
    ).eval()

    postprocessing = nn.Sigmoid()
    path = tmp_path / "graph_representation.ptc"

    graph = make_graph()

    # Current exporter validates these deployment fields.
    graph["node_attrs"] = torch.ones(
        graph["positions"].shape[0],
        1,
        dtype=graph["positions"].dtype,
    )
    graph["shifts"] = graph[
        "unit_shifts"
    ].clone()

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
