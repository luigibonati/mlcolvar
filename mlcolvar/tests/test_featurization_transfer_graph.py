from __future__ import annotations

from typing import Dict, Optional

import pytest
import torch
from torch import nn

from mlcolvar.core import BaseGNN
from mlcolvar.data import DictDataset
from mlcolvar.featurization.transfer import (
    TransferFeaturizer,
    TransferModel,
    export_transfer_torchscript,
    precompute_committor_cache,
)
from torch_geometric.data import Data


class GraphShiftPreprocessing(nn.Module):
    """Shift graph positions while preserving all other graph tensors."""

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
            data["positions"]
            + shift
        )

        return output


class DummyGraphEncoder(BaseGNN):
    """Small graph-level encoder satisfying the BaseGNN transfer contract."""

    def __init__(
        self,
        pooling_operation: Optional[str] = "mean",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        # Use the real BaseGNN constructor. ``in_features`` and
        # ``out_features`` are read-only properties of BaseGNN and must not
        # be assigned directly.
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

        # Keep all floating-point BaseGNN metadata and helper modules in the
        # same precision as the dummy encoder.
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
    """Minimal graph-based pretrained CV."""

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
    """Minimal tensor CV used to test tensor/graph type validation."""

    def __init__(self) -> None:
        super().__init__()

        self.in_features = 3
        self.out_features = 2
        self.nn = nn.Linear(
            3,
            2,
        )

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


def test_graph_transfer_featurizer_features_metadata_and_dtype() -> None:
    """Extract graph features and copy the pretrained GNN metadata."""

    pretrained_model = DummyGraphCV()

    featurizer = TransferFeaturizer(
        model=pretrained_model,
        freeze=True,
    )

    data = make_graph()

    output = featurizer(data)

    expected = torch.tensor(
        [
            [4.0, 3.0],
            [4.0, 6.0],
        ],
        dtype=torch.float32,
    )

    assert featurizer.in_features is None
    assert featurizer.out_features == 2
    assert featurizer.pooling_operation == "mean"

    torch.testing.assert_close(
        featurizer.cutoff,
        torch.tensor(4.0),
    )
    torch.testing.assert_close(
        featurizer.buffer,
        torch.tensor(0.5),
    )
    torch.testing.assert_close(
        featurizer.long_range_cutoff,
        torch.tensor(-1.0),
    )

    assert featurizer.atomic_numbers.tolist() == [
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

    # The input dictionary itself is not replaced or modified in-place.
    assert data["positions"].dtype == torch.float64

    torch.testing.assert_close(
        data["positions"],
        make_graph()["positions"],
    )


def test_frozen_graph_featurizer_preserves_position_gradients() -> None:
    """Keep coordinate gradients while freezing the pretrained graph CV."""

    pretrained_model = DummyGraphCV()

    featurizer = TransferFeaturizer(
        model=pretrained_model,
        freeze=True,
    )

    data = make_graph()
    data["positions"].requires_grad_(True)

    output = featurizer(data)
    output.sum().backward()

    gradient = data["positions"].grad

    assert gradient is not None
    assert gradient.shape == data["positions"].shape
    assert torch.isfinite(gradient).all()
    assert torch.count_nonzero(gradient) > 0

    assert all(
        not parameter.requires_grad
        for parameter in pretrained_model.parameters()
    )

    assert all(
        parameter.grad is None
        for parameter in pretrained_model.parameters()
    )

    featurizer.train()

    assert featurizer.training
    assert not pretrained_model.training
    assert not pretrained_model.nn.training


def test_graph_featurizer_state_dict_contains_model_and_metadata() -> None:
    """Persist the graph encoder and copied deployment metadata."""

    featurizer = TransferFeaturizer(
        model=DummyGraphCV(),
        freeze=True,
    )

    state = featurizer.state_dict()

    assert "model.nn.scale" in state
    assert "cutoff" in state
    assert "buffer" in state
    assert "long_range_cutoff" in state
    assert "atomic_numbers" in state
    assert "_model_reference" not in state


def test_graph_transfer_model_is_basegnn_and_trains_only_readout() -> None:
    """Wrap frozen graph features in a trainable graph-compatible readout."""

    pretrained_model = DummyGraphCV()

    featurizer = TransferFeaturizer(
        model=pretrained_model,
        freeze=True,
    )

    model = TransferModel(
        featurizer=featurizer,
        n_out=1,
        hidden_layers=(),
    )

    assert isinstance(
        model,
        BaseGNN,
    )

    assert model.featurizer is featurizer
    assert model.out_features == 1

    assert "_radial_embedding" not in model._modules

    torch.testing.assert_close(
        model.cutoff,
        featurizer.cutoff,
    )
    torch.testing.assert_close(
        model.buffer,
        featurizer.buffer,
    )
    torch.testing.assert_close(
        model.long_range_cutoff,
        featurizer.long_range_cutoff,
    )

    assert model.atomic_numbers.tolist() == [
        1,
        6,
        8,
    ]

    assert all(
        not parameter.requires_grad
        for parameter in pretrained_model.parameters()
    )

    trainable_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    # Linear(2, 1): two weights and one bias.
    assert sum(
        parameter.numel()
        for parameter in trainable_parameters
    ) == 3

    linear = next(
        module
        for module in model.readout.modules()
        if isinstance(module, nn.Linear)
    )

    with torch.no_grad():
        linear.weight.fill_(1.0)
        linear.bias.zero_()

    data = make_graph()
    data["positions"].requires_grad_(True)

    output = model(data)

    assert output.shape == (2, 1)
    assert output.dtype == linear.weight.dtype

    output.sum().backward()

    assert data["positions"].grad is not None
    assert torch.isfinite(
        data["positions"].grad
    ).all()
    assert torch.count_nonzero(
        data["positions"].grad
    ) > 0

    assert linear.weight.grad is not None
    assert pretrained_model.nn.scale.grad is None

    model.train()

    assert model.training
    assert model.featurizer.training
    assert not pretrained_model.training
    assert not pretrained_model.nn.training


def test_graph_transfer_model_can_be_traced() -> None:
    """Trace the graph transfer encoder and readout as one module."""

    model = TransferModel(
        featurizer=TransferFeaturizer(
            model=DummyGraphCV(),
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


def test_graph_featurizer_requires_graph_level_pooling() -> None:
    """Reject node-level encoders for graph-level transfer targets."""

    with pytest.raises(
        ValueError,
        match="must use graph-level pooling",
    ):
        TransferFeaturizer(
            model=DummyGraphCV(
                pooling_operation=None,
            )
        )


def test_graph_featurizer_rejects_tensor_norm_in() -> None:
    """Reject tensor normalization applied directly to graph dictionaries."""

    featurizer = TransferFeaturizer(
        model=DummyGraphCV(
            use_norm_in=True,
        )
    )

    with pytest.raises(
        ValueError,
        match="Input normalization is tensor-based",
    ):
        featurizer(
            make_graph()
        )


def test_transfer_api_dispatches_tensor_and_graph_models() -> None:
    """The public factories select tensor or graph implementations automatically."""

    tensor_featurizer = TransferFeaturizer(
        model=DummyTensorCV(),
        mode="latent",
    )

    graph_featurizer = TransferFeaturizer(
        model=DummyGraphCV(),
        mode="latent",
    )

    tensor_model = TransferModel(
        featurizer=tensor_featurizer,
        n_out=1,
        hidden_layers=(),
    )

    graph_model = TransferModel(
        featurizer=graph_featurizer,
        n_out=1,
        hidden_layers=(),
    )

    assert tensor_featurizer.in_features == 3
    assert graph_featurizer.in_features is None

    assert not isinstance(tensor_model, BaseGNN)
    assert isinstance(graph_model, BaseGNN)

    with pytest.raises(
        ValueError,
        match="support only.*latent",
    ):
        TransferFeaturizer(
            model=DummyGraphCV(),
            mode="output",
        )


@pytest.mark.parametrize(
    "hidden_layers",
    [
        (0,),
        (-2,),
        (8, 0),
    ],
)
def test_graph_transfer_model_rejects_invalid_hidden_layers(
    hidden_layers,
) -> None:
    featurizer = TransferFeaturizer(
        model=DummyGraphCV(),
    )

    with pytest.raises(
        ValueError,
        match="positive integers",
    ):
        TransferModel(
            featurizer=featurizer,
            hidden_layers=hidden_layers,
        )


def make_graph_sample(
    offset: float,
    label: float = 2.0,
) -> Data:
    """Create one fixed-composition graph for cache tests."""

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

        # Explicitly preserve the isolated third atom. Without this, PyG may
        # infer only two nodes from edge_index and create a too-short batch.
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


def test_graph_transfer_raw_and_cached_paths_agree() -> None:
    """Graph inference and cached latent inference must share one head."""

    featurizer = TransferFeaturizer(
        model=DummyGraphCV(),
        freeze=True,
    )

    model = TransferModel(
        featurizer=featurizer,
        n_out=1,
        hidden_layers=(),
    ).eval()

    data = make_graph()

    with torch.no_grad():
        latent = featurizer(data)
        output_raw = model.forward_raw(data)
        output_cached = model.forward_features(latent)

    assert model.raw_in_features is None
    assert model.latent_features == 2
    assert model.transfer_out_features == 1

    torch.testing.assert_close(
        output_raw,
        output_cached,
    )


def test_graph_cache_matches_direct_features() -> None:
    """Cache graph-level latent values and dense coordinate Jacobians."""

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

    featurizer = TransferFeaturizer(
        model=DummyGraphCV(
            use_preprocessing=False,
        ),
        freeze=True,
    )

    cached_dataset, cached_derivatives = (
        precompute_committor_cache(
            featurizer=featurizer,
            dataset=dataset,
            batch_size=1,
            device="cpu",
            output_device="cpu",
            separate_boundary_dataset=False,
        )
    )

    graph_batch = dataset.get_graph_inputs()

    with torch.no_grad():
        expected_latent = featurizer(
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

    # Mean pooling over three atoms:
    # h_0 = mean(2*x), h_1 = mean(2*y).
    expected_jacobian[:, :, 0, 0] = 2.0 / 3.0
    expected_jacobian[:, :, 1, 1] = 2.0 / 3.0

    torch.testing.assert_close(
        cached_derivatives.jacobian,
        expected_jacobian,
    )


def test_graph_transfer_torchscript_roundtrip(
    tmp_path,
) -> None:
    """Export and reload a complete graph-input transfer model."""

    model = TransferModel(
        featurizer=TransferFeaturizer(
            model=DummyGraphCV(),
            freeze=True,
        ),
        n_out=1,
        hidden_layers=(),
    ).eval()

    postprocessing = nn.Sigmoid()
    path = tmp_path / "graph_transfer.ptc"

    graph = make_graph()

    # The current graph exporter validates these deployment fields.
    graph["node_attrs"] = torch.ones(
        graph["positions"].shape[0],
        1,
        dtype=graph["positions"].dtype,
    )
    graph["shifts"] = graph[
        "unit_shifts"
    ].clone()

    export_transfer_torchscript(
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
            model.forward_raw(graph)
        )
        output = loaded(graph)

    assert path.exists()

    torch.testing.assert_close(
        output,
        expected,
        rtol=1e-5,
        atol=1e-6,
    )


def test_graph_cache_restores_featurizer_state() -> None:
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

    featurizer = TransferFeaturizer(
        model=DummyGraphCV(
            use_preprocessing=False,
        ),
        freeze=True,
    )

    featurizer.train()
    original_device = featurizer._model_reference.device

    precompute_committor_cache(
        featurizer=featurizer,
        dataset=dataset,
        batch_size=1,
        device="cpu",
        output_device="cpu",
        separate_boundary_dataset=False,
    )

    assert featurizer.training
    assert featurizer._model_reference.device == original_device
    assert not featurizer.model.training