from typing import Any

import torch
from torch import nn

from mlcolvar.core import BaseGNN


def _as_positive_int(
    value: Any,
    name: str,
) -> int:
    """Convert a scalar value to a positive Python integer."""

    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError(
                f"`{name}` must be scalar."
            )

        value = value.detach().cpu().item()

    value = int(value)

    if value <= 0:
        raise ValueError(
            f"`{name}` must be positive. Found {value}."
        )

    return value


def _module_reference_tensor(
    module: nn.Module,
) -> torch.Tensor:
    """Return a scalar matching a module's floating dtype/device."""

    for parameter in module.parameters():
        if (
            parameter.is_floating_point()
            or parameter.is_complex()
        ):
            return torch.empty(
                (),
                dtype=parameter.dtype,
                device=parameter.device,
            )

    for buffer in module.buffers():
        if (
            buffer.is_floating_point()
            or buffer.is_complex()
        ):
            return torch.empty(
                (),
                dtype=buffer.dtype,
                device=buffer.device,
            )

    return torch.empty(())


def _infer_model_output_dimension(
    model: nn.Module,
) -> int:
    """Infer the output dimension of a model or its internal ``nn`` block."""

    internal_model = getattr(
        model,
        "nn",
        None,
    )

    if internal_model is None:
        internal_model = model

    out_features = getattr(
        internal_model,
        "out_features",
        None,
    )

    if out_features is not None:
        return _as_positive_int(
            out_features,
            "out_features",
        )

    n_out = getattr(
        internal_model,
        "n_out",
        None,
    )

    if n_out is not None:
        return _as_positive_int(
            n_out,
            "n_out",
        )

    raise ValueError(
        "Cannot infer the feature dimension from "
        f"{model.__class__.__name__}. "
        "Pass `out_features` explicitly."
    )


def _is_graph_model(
    model: nn.Module,
) -> bool:
    """Return whether a pretrained mlcolvar CV contains a BaseGNN."""

    return isinstance(
        getattr(model, "nn", None),
        BaseGNN,
    )


def _get_graph_encoder(
    model: nn.Module,
) -> BaseGNN:
    """Return the BaseGNN encoder contained in a pretrained mlcolvar CV."""

    encoder = getattr(
        model,
        "nn",
        None,
    )

    if not isinstance(
        encoder,
        BaseGNN,
    ):
        raise TypeError(
            f"{model.__class__.__name__} is not graph-based: "
            f"expected `.nn` to be a BaseGNN, found {type(encoder)}."
        )

    return encoder