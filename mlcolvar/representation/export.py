from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path

import torch
from torch import nn

from mlcolvar.core import BaseGNN


__all__ = ["export_representation_torchscript"]


class _RepresentationInferenceModel(nn.Module):
    """Apply optional postprocessing to a representation model."""

    def __init__(self, model: nn.Module, postprocessing: nn.Module | None = None):
        super().__init__()
        self.model = model
        self.postprocessing = postprocessing or nn.Identity()

    def forward(self, x, cell=None):
        return self.postprocessing(self.model(x, cell=cell))


def _enable_lightning_jit(module):
    for child in module.modules():
        if hasattr(child, "_jit_is_scripting"):
            child._jit_is_scripting = True


def _prepare_vector_example(model, example_input, dtype):
    in_features = int(model.in_features)

    if example_input is None:
        return torch.zeros(1, in_features, dtype=dtype)
    if not torch.is_tensor(example_input):
        raise TypeError("Vector models require a tensor `example_input`.")

    example_input = example_input.detach().cpu().to(dtype=dtype)
    if example_input.ndim < 2 or example_input.shape[-1] != in_features:
        raise ValueError(
            f"Expected example input shape (..., {in_features}); "
            f"found {tuple(example_input.shape)}."
        )
    return example_input


def _prepare_graph_example(example_input, dtype):
    if example_input is None:
        raise ValueError("Graph export requires an explicit `example_input`.")

    if hasattr(example_input, "to_dict"):
        example_input = example_input.to_dict()
    if not isinstance(example_input, Mapping):
        raise TypeError(
            "Graph `example_input` must be a mapping or expose `to_dict()`."
        )

    graph = {}
    for key, value in example_input.items():
        if torch.is_tensor(value):
            value = value.detach().cpu()
            if value.is_floating_point() or value.is_complex():
                value = value.to(dtype=dtype)
            graph[str(key)] = value

    required = {"positions", "node_attrs", "edge_index", "shifts", "batch"}
    missing = sorted(required - graph.keys())
    if missing:
        raise KeyError(
            f"Graph example is missing required tensor keys: {missing}."
        )

    return graph


def export_representation_torchscript(
    model: nn.Module,
    path: str | Path,
    *,
    postprocessing: nn.Module | None = None,
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
        else _prepare_vector_example(model, example_input, dtype)
    )

    inference = _RepresentationInferenceModel(
        deepcopy(model),
        deepcopy(postprocessing) if postprocessing is not None else None,
    )
    inference.to(device="cpu", dtype=dtype).eval().requires_grad_(False)
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