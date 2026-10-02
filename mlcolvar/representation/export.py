from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path

import torch
from torch import nn

from ._utils import module_reference_tensor
from .base import Representation

__all__ = ["export_representation_torchscript"]


class _RepresentationInferenceModel(nn.Module):
    """Apply optional postprocessing to a representation."""

    def __init__(
        self,
        representation: Representation,
        postprocessing: nn.Module | None = None,
    ):
        super().__init__()
        self.representation = representation
        self.postprocessing = postprocessing or nn.Identity()

    def forward(self, x, cell=None):
        return self.postprocessing(
            self.representation(x, cell=cell)
        )


def _enable_lightning_jit(module):
    for child in module.modules():
        if hasattr(child, "_jit_is_scripting"):
            child._jit_is_scripting = True


def _prepare_vector_example(
    representation,
    example_input,
    dtype,
):
    in_features = int(representation.in_features)

    if example_input is None:
        return torch.zeros(
            1,
            in_features,
            dtype=dtype,
        )

    if not torch.is_tensor(example_input):
        raise TypeError(
            "Vector representations require a tensor `example_input`."
        )

    example_input = (
        example_input
        .detach()
        .cpu()
        .to(dtype=dtype)
    )

    if (
        example_input.ndim < 2
        or example_input.shape[-1] != in_features
    ):
        raise ValueError(
            f"Expected example input shape (..., {in_features}); "
            f"found {tuple(example_input.shape)}."
        )

    return example_input


def _prepare_graph_example(
    example_input,
    dtype,
):
    if example_input is None:
        raise ValueError(
            "Graph export requires an explicit `example_input`."
        )

    if hasattr(example_input, "to_dict"):
        example_input = example_input.to_dict()

    if not isinstance(example_input, Mapping):
        raise TypeError(
            "Graph `example_input` must be a mapping "
            "or expose `to_dict()`."
        )

    graph = {}

    for key, value in example_input.items():
        if torch.is_tensor(value):
            value = value.detach().cpu()

            if value.is_floating_point() or value.is_complex():
                value = value.to(dtype=dtype)

            graph[str(key)] = value

    if not graph:
        raise ValueError(
            "Graph `example_input` must contain tensor fields."
        )

    return graph


def export_representation_torchscript(
    representation: Representation,
    path: str | Path,
    *,
    postprocessing: nn.Module | None = None,
    example_input=None,
    dtype: torch.dtype | None = None,
    freeze: bool = True,
    check_trace: bool = True,
) -> torch.jit.ScriptModule:
    """Trace and save a reusable representation.

    Parameters
    ----------
    representation : Representation
        Representation to export.
    path : str or pathlib.Path
        Output path of the TorchScript model.
    postprocessing : torch.nn.Module, optional
        Optional module applied to the representation output.
    example_input
        Example input used for tracing. Required for graph representations.
        For vector representations, a zero tensor is generated automatically
        when not provided.
    dtype : torch.dtype, optional
        Floating-point dtype used during tracing. By default, use the
        representation dtype.
    freeze : bool, default=True
        If True, freeze the traced TorchScript module.
    check_trace : bool, default=True
        Whether to check the traced graph against the example input.

    Returns
    -------
    torch.jit.ScriptModule
        Exported TorchScript module.
    """
    if not isinstance(
        representation,
        Representation,
    ):
        raise TypeError(
            "`representation` must derive from `Representation`."
        )

    if dtype is None:
        dtype = module_reference_tensor(
            representation
        ).dtype

    if representation.input_kind == "graph":
        prepared = _prepare_graph_example(
            example_input,
            dtype,
        )
    else:
        prepared = _prepare_vector_example(
            representation,
            example_input,
            dtype,
        )

    inference = _RepresentationInferenceModel(
        deepcopy(representation),
        deepcopy(postprocessing)
        if postprocessing is not None
        else None,
    )

    inference.to(
        device="cpu",
        dtype=dtype,
    ).eval().requires_grad_(False)

    prepare = getattr(
        inference.representation,
        "prepare_for_torchscript",
        None,
    )
    if callable(prepare):
        prepare()

    _enable_lightning_jit(
        inference
    )

    with torch.no_grad():
        traced = torch.jit.trace(
            inference,
            (prepared,),
            strict=False,
            check_trace=check_trace,
        )

        if freeze:
            traced = torch.jit.freeze(
                traced
            )

    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.jit.save(
        traced,
        str(path),
    )

    return traced