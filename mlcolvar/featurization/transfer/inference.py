from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Dict, Optional, Union

import torch
from torch import nn

from .readouts import (
    _GraphTransferModel,
    _TensorTransferModel,
)


__all__ = [
    "TransferInferenceModel",
    "export_transfer_torchscript",
]


class _TensorTransferInferenceModel(nn.Module):
    """Raw-tensor-input transfer model for inference."""

    def __init__(
        self,
        model: _TensorTransferModel,
        postprocessing: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()

        self.featurizer = model.featurizer
        self.head = model.nn
        self.postprocessing = (
            postprocessing
            if postprocessing is not None
            else nn.Identity()
        )

        self.in_features = int(model.raw_in_features)
        self.latent_features = int(model.latent_features)
        self.out_features = int(model.out_features)

    def forward_features(
        self,
        features: torch.Tensor,
    ) -> torch.Tensor:
        output = self.head(features)
        return self.postprocessing(output)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        features = self.featurizer(x)
        return self.forward_features(features)


class _GraphTransferInferenceModel(nn.Module):
    """Raw-graph-input transfer model for inference."""

    def __init__(
        self,
        model: _GraphTransferModel,
        postprocessing: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()

        self.featurizer = model.featurizer
        self.head = model.readout
        self.postprocessing = (
            postprocessing
            if postprocessing is not None
            else nn.Identity()
        )

        self.latent_features = int(model.latent_features)
        self.out_features = int(model.transfer_out_features)

    def forward_features(
        self,
        features: torch.Tensor,
    ) -> torch.Tensor:
        output = self.head(features)
        return self.postprocessing(output)

    def forward(
        self,
        data: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        features = self.featurizer(data)
        return self.forward_features(features)


def TransferInferenceModel(
    model: nn.Module,
    postprocessing: Optional[nn.Module] = None,
) -> nn.Module:
    """Create an inference wrapper for a transfer model."""

    if isinstance(model, _TensorTransferModel):
        return _TensorTransferInferenceModel(
            model=model,
            postprocessing=postprocessing,
        )

    if isinstance(model, _GraphTransferModel):
        return _GraphTransferInferenceModel(
            model=model,
            postprocessing=postprocessing,
        )

    raise TypeError(
        "`model` must be created with `TransferModel`. "
        f"Found {type(model)}."
    )


def _enable_lightning_jit(
    module: nn.Module,
) -> None:
    """Allow nested Lightning modules to be inspected by TorchScript."""

    for submodule in module.modules():
        if hasattr(submodule, "_jit_is_scripting"):
            submodule._jit_is_scripting = True


def _prepare_tensor_example(
    model: _TensorTransferModel,
    example_input: Optional[torch.Tensor],
    dtype: torch.dtype,
) -> torch.Tensor:
    """Prepare an example input for tensor tracing."""

    input_dim = int(model.raw_in_features)

    if example_input is None:
        return torch.zeros(
            1,
            input_dim,
            dtype=dtype,
            device="cpu",
        )

    if not torch.is_tensor(example_input):
        raise TypeError(
            "Tensor transfer models require `example_input` "
            "to be a tensor."
        )

    example_input = example_input.detach().to(
        device="cpu",
        dtype=dtype,
    )

    if example_input.ndim < 2:
        raise ValueError(
            "`example_input` must include a batch dimension."
        )

    if example_input.shape[-1] != input_dim:
        raise ValueError(
            f"Expected input dimension {input_dim}, "
            f"found {example_input.shape[-1]}."
        )

    return example_input


def _prepare_graph_example(
    example_input,
    dtype: torch.dtype,
) -> Dict[str, torch.Tensor]:
    """Prepare an example graph for tracing."""

    if example_input is None:
        raise ValueError(
            "Graph export requires an explicit `example_input`."
        )

    if hasattr(example_input, "to_dict"):
        example_input = example_input.to_dict()

    if not isinstance(example_input, Mapping):
        raise TypeError(
            "Graph `example_input` must be a tensor dictionary "
            "or an object implementing `to_dict()`."
        )

    graph: Dict[str, torch.Tensor] = {}

    for key, value in example_input.items():
        if not torch.is_tensor(value):
            continue

        value = value.detach().cpu()

        if value.is_floating_point() or value.is_complex():
            value = value.to(dtype=dtype)

        graph[str(key)] = value

    required_keys = {
        "positions",
        "node_attrs",
        "edge_index",
        "shifts",
        "batch",
    }

    missing = sorted(
        required_keys.difference(graph)
    )

    if missing:
        raise KeyError(
            "Graph example is missing required tensor keys: "
            f"{missing}."
        )

    return graph


def export_transfer_torchscript(
    model: nn.Module,
    path: Union[str, Path],
    postprocessing: Optional[nn.Module] = None,
    example_input=None,
    dtype: torch.dtype = torch.float32,
    freeze: bool = True,
    check_trace: bool = True,
) -> torch.jit.ScriptModule:
    """Export a complete transfer model as TorchScript.

    Tensor models accept raw descriptor tensors. Graph models accept
    ``Dict[str, Tensor]`` and require an explicit example graph.
    """

    model_copy = deepcopy(model)

    postprocessing_copy = (
        deepcopy(postprocessing)
        if postprocessing is not None
        else None
    )

    inference_model = TransferInferenceModel(
        model=model_copy,
        postprocessing=postprocessing_copy,
    )

    if isinstance(model, _TensorTransferModel):
        prepared_input = _prepare_tensor_example(
            model=model,
            example_input=example_input,
            dtype=dtype,
        )

    elif isinstance(model, _GraphTransferModel):
        prepared_input = _prepare_graph_example(
            example_input=example_input,
            dtype=dtype,
        )

    else:
        raise TypeError(
            "`model` must be created with `TransferModel`. "
            f"Found {type(model)}."
        )

    inference_model = inference_model.to(
        device="cpu",
        dtype=dtype,
    ).eval()

    inference_model.requires_grad_(False)

    _enable_lightning_jit(inference_model)

    with torch.no_grad():
        traced_model = torch.jit.trace(
            inference_model,
            (prepared_input,),
            strict=False,
            check_trace=check_trace,
        )

        if freeze:
            traced_model = torch.jit.freeze(
                traced_model
            )

    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.jit.save(
        traced_model,
        str(path),
    )

    return traced_model