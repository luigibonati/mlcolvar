from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Dict, Optional, Union

import torch
from torch import nn

from mlcolvar.core import BaseGNN


__all__ = [
    "RepresentationInferenceModel",
    "export_representation_torchscript",
]


class RepresentationInferenceModel(nn.Module):
    """Apply optional postprocessing to a representation model."""

    def __init__(
        self,
        model: nn.Module,
        postprocessing: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.postprocessing = (
            nn.Identity()
            if postprocessing is None
            else postprocessing
        )

    def forward(
        self,
        x: torch.Tensor,
        cell: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.postprocessing(
            self.model(x, cell=cell)
        )


def _enable_lightning_jit(module: nn.Module) -> None:
    for child in module.modules():
        if hasattr(child, "_jit_is_scripting"):
            child._jit_is_scripting = True


def _prepare_tensor_example(
    model: nn.Module,
    example_input: Optional[torch.Tensor],
    dtype: torch.dtype,
) -> torch.Tensor:
    input_dim = int(model.in_features)
    if example_input is None:
        return torch.zeros(1, input_dim, dtype=dtype)
    if not torch.is_tensor(example_input):
        raise TypeError("Tensor models require a tensor `example_input`.")

    example_input = example_input.detach().cpu().to(dtype=dtype)
    if example_input.ndim < 2 or example_input.shape[-1] != input_dim:
        raise ValueError(
            f"Expected example input shape (..., {input_dim}); "
            f"found {tuple(example_input.shape)}."
        )
    return example_input


def _prepare_graph_example(example_input, dtype: torch.dtype) -> Dict[str, torch.Tensor]:
    if example_input is None:
        raise ValueError("Graph export requires an explicit `example_input`.")
    if hasattr(example_input, "to_dict"):
        example_input = example_input.to_dict()
    if not isinstance(example_input, Mapping):
        raise TypeError("Graph `example_input` must be a mapping or expose `to_dict()`." )

    graph: Dict[str, torch.Tensor] = {}
    for key, value in example_input.items():
        if not torch.is_tensor(value):
            continue
        value = value.detach().cpu()
        if value.is_floating_point() or value.is_complex():
            value = value.to(dtype=dtype)
        graph[str(key)] = value

    required = {"positions", "node_attrs", "edge_index", "shifts", "batch"}
    missing = sorted(required.difference(graph))
    if missing:
        raise KeyError(f"Graph example is missing required tensor keys: {missing}.")
    return graph


def export_representation_torchscript(
    model: nn.Module,
    path: Union[str, Path],
    *,
    postprocessing: Optional[nn.Module] = None,
    example_input=None,
    dtype: torch.dtype = torch.float32,
    freeze: bool = True,
    check_trace: bool = True,
) -> torch.jit.ScriptModule:
    """Trace and save a complete representation model."""
    is_graph = isinstance(model, BaseGNN) or getattr(model, "in_features", 1) is None
    prepared = (
        _prepare_graph_example(example_input, dtype)
        if is_graph
        else _prepare_tensor_example(model, example_input, dtype)
    )

    inference = RepresentationInferenceModel(
        deepcopy(model),
        deepcopy(postprocessing) if postprocessing is not None else None,
    ).to(device="cpu", dtype=dtype).eval()
    inference.requires_grad_(False)
    _enable_lightning_jit(inference)

    with torch.no_grad():
        traced = torch.jit.trace(
            inference,
            (prepared,),
            strict=False,
            check_trace=check_trace,
        )
        if freeze:
            traced = torch.jit.freeze(traced)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.jit.save(traced, str(path))
    return traced
