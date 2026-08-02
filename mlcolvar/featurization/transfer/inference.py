from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Dict, Optional, Union

import torch
from torch import nn

from .readouts import (
    CVGraphReadoutModel,
    CVReadoutModel,
)


__all__ = [
    "CVTransferInferenceModel",
    "CVGraphTransferInferenceModel",
    "export_transfer_torchscript",
]


class CVTransferInferenceModel(nn.Module):
    """Complete raw-tensor-input transfer model.

    Pipeline
    --------
    raw tensor -> frozen featurizer -> trained readout -> postprocessing
    """

    def __init__(
        self,
        readout: CVReadoutModel,
        postprocessing: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()

        if not isinstance(readout, CVReadoutModel):
            raise TypeError(
                "`readout` must be a CVReadoutModel. "
                f"Found {type(readout)}."
            )

        self.featurizer = readout.featurizer
        self.head = readout.nn
        self.postprocessing = (
            postprocessing
            if postprocessing is not None
            else nn.Identity()
        )

        self.in_features = int(
            getattr(
                readout,
                "raw_in_features",
                readout.featurizer.in_features,
            )
        )
        self.latent_features = int(
            getattr(
                readout,
                "latent_features",
                readout.featurizer.out_features,
            )
        )
        self.out_features = int(readout.out_features)

    def forward_features(
        self,
        features: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate the trained head from latent features."""
        output = self.head(features)
        return self.postprocessing(output)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate the complete model from raw tensor input."""
        features = self.featurizer(x)
        return self.forward_features(features)


class CVGraphTransferInferenceModel(nn.Module):
    """Complete raw-graph-input transfer model.

    Pipeline
    --------
    graph dictionary -> frozen graph featurizer
    -> trained readout -> postprocessing
    """

    def __init__(
        self,
        readout: CVGraphReadoutModel,
        postprocessing: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()

        if not isinstance(readout, CVGraphReadoutModel):
            raise TypeError(
                "`readout` must be a CVGraphReadoutModel. "
                f"Found {type(readout)}."
            )

        self.featurizer = readout.featurizer
        self.head = readout.readout
        self.postprocessing = (
            postprocessing
            if postprocessing is not None
            else nn.Identity()
        )

        self.latent_features = int(
            getattr(
                readout,
                "latent_features",
                readout.featurizer.out_features,
            )
        )
        self.out_features = int(readout.out_features)

    def forward_features(
        self,
        features: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate the trained head from graph latent features."""
        output = self.head(features)
        return self.postprocessing(output)

    def forward(
        self,
        data: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Evaluate the complete model from a graph tensor dictionary."""
        features = self.featurizer(data)
        return self.forward_features(features)


def _enable_lightning_jit(
    module: nn.Module,
) -> None:
    """Allow nested Lightning modules to be inspected by TorchScript."""

    for submodule in module.modules():
        if hasattr(submodule, "_jit_is_scripting"):
            submodule._jit_is_scripting = True


def _prepare_tensor_example(
    readout: CVReadoutModel,
    example_input: Optional[torch.Tensor],
    dtype: torch.dtype,
) -> torch.Tensor:
    """Prepare the tracing input for a tensor readout."""

    raw_input_dim = int(
        getattr(
            readout,
            "raw_in_features",
            readout.featurizer.in_features,
        )
    )

    if example_input is None:
        return torch.zeros(
            1,
            raw_input_dim,
            dtype=dtype,
            device="cpu",
        )

    if not torch.is_tensor(example_input):
        raise TypeError(
            "Tensor readouts require `example_input` to be a tensor."
        )

    example_input = example_input.detach().to(
        device="cpu",
        dtype=dtype,
    )

    if example_input.ndim < 2:
        raise ValueError(
            "`example_input` must include a batch dimension."
        )

    if example_input.shape[-1] != raw_input_dim:
        raise ValueError(
            f"Expected raw input dimension {raw_input_dim}, "
            f"found {example_input.shape[-1]}."
        )

    return example_input


def _prepare_graph_example(
    example_input,
    dtype: torch.dtype,
) -> Dict[str, torch.Tensor]:
    """Convert a PyG graph or tensor dictionary into a tracing input."""

    if example_input is None:
        raise ValueError(
            "Graph export requires an explicit `example_input`."
        )

    if hasattr(example_input, "to_dict"):
        example_input = example_input.to_dict()

    if not isinstance(example_input, Mapping):
        raise TypeError(
            "Graph `example_input` must be a tensor dictionary or "
            "an object implementing `to_dict()`."
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

    missing_keys = sorted(
        required_keys.difference(graph)
    )

    if missing_keys:
        raise KeyError(
            "Graph example is missing required tensor keys: "
            f"{missing_keys}."
        )

    return graph


def export_transfer_torchscript(
    readout: Union[
        CVReadoutModel,
        CVGraphReadoutModel,
    ],
    path: Union[str, Path],
    postprocessing: Optional[nn.Module] = None,
    example_input=None,
    dtype: torch.dtype = torch.float32,
    freeze: bool = True,
    check_trace: bool = True,
) -> torch.jit.ScriptModule:
    """Export a complete raw-input transfer model as TorchScript.

    Tensor models accept raw descriptor tensors. Graph models accept
    ``Dict[str, Tensor]`` and require an explicit graph example input.
    """

    readout_copy = deepcopy(readout)
    postprocessing_copy = (
        deepcopy(postprocessing)
        if postprocessing is not None
        else None
    )

    if isinstance(readout, CVReadoutModel):
        inference_model: nn.Module = CVTransferInferenceModel(
            readout=readout_copy,
            postprocessing=postprocessing_copy,
        )

        prepared_input = _prepare_tensor_example(
            readout=readout,
            example_input=example_input,
            dtype=dtype,
        )

    elif isinstance(readout, CVGraphReadoutModel):
        inference_model = CVGraphTransferInferenceModel(
            readout=readout_copy,
            postprocessing=postprocessing_copy,
        )

        prepared_input = _prepare_graph_example(
            example_input=example_input,
            dtype=dtype,
        )

    else:
        raise TypeError(
            "`readout` must be a CVReadoutModel or "
            "CVGraphReadoutModel. "
            f"Found {type(readout)}."
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