from typing import Dict, Optional

import torch
from torch import nn

from mlcolvar.core import BaseGNN

from ..base import GraphRepresentation, Representation, VectorRepresentation
from .._utils import as_positive_int, module_reference_tensor
from ._utils import to_float, to_int_list


__all__ = ["MLColvarRepresentation"]


def _infer_model_output_dimension(model: nn.Module) -> int:
    """Infer the latent output dimension of an mlcolvar model."""
    internal = getattr(model, "nn", model)

    for name in ("out_features", "n_out"):
        value = getattr(internal, name, None)
        if value is not None:
            return as_positive_int(value, name)

    raise ValueError(
        "Cannot infer the representation dimension from "
        f"{model.__class__.__name__}; pass `out_features` explicitly."
    )


class _FrozenModelMixin:
    def _init_model(self, model: nn.Module) -> None:
        self.model = model
        self.register_buffer(
            "_model_reference",
            module_reference_tensor(model),
            persistent=False,
        )
        self._freeze_module(model)

    def _apply_preprocessing(self, data, cell=None):
        preprocessing = getattr(self.model, "preprocessing", None)
        if preprocessing is None:
            return data
        return preprocessing(data) if cell is None else preprocessing(data, cell=cell)

    def _validate_output(self, output):
        if output.shape[-1] != self.out_features:
            raise ValueError(
                f"Expected {self.out_features} latent features, "
                f"found {output.shape[-1]}."
            )
        return output


class _VectorMLColvarRepresentation(
    _FrozenModelMixin,
    VectorRepresentation,
):
    """Representation adapter for vector-based mlcolvar models."""

    __constants__ = ["mode"]

    def __init__(
        self,
        model: nn.Module,
        *,
        mode: str,
        out_features: Optional[int],
        freeze: bool,
    ) -> None:
        if getattr(model, "in_features", None) is None:
            raise TypeError(f"{model.__class__.__name__} is graph-based.")

        if mode == "latent":
            if getattr(model, "nn", None) is None:
                raise TypeError(
                    f"{model.__class__.__name__} does not expose "
                    "an `.nn` latent encoder."
                )
            resolved_out = (
                _infer_model_output_dimension(model)
                if out_features is None
                else as_positive_int(out_features, "out_features")
            )
        else:
            if out_features is not None:
                raise ValueError(
                    "`out_features` is only valid with mode='latent'."
                )
            if mode == "forward" and not hasattr(model, "forward_cv"):
                raise TypeError(
                    f"{model.__class__.__name__} does not implement "
                    "`forward_cv()`."
                )
            resolved_out = as_positive_int(
                model.out_features,
                "model.out_features",
            )

        VectorRepresentation.__init__(
            self,
            in_features=as_positive_int(
                model.in_features,
                "model.in_features",
            ),
            out_features=resolved_out,
            output_kind="system",
            freeze=freeze,
        )

        self.mode = mode
        self._init_model(model)

    def forward(
        self,
        x: torch.Tensor,
        cell: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if x.ndim < 2:
            raise ValueError("`x` must include a batch dimension.")
        if x.shape[-1] != self.in_features:
            raise ValueError(
                f"Expected {self.in_features} input features, "
                f"found {x.shape[-1]}."
            )

        x = x.to(self._model_reference)
        if cell is not None:
            cell = cell.to(self._model_reference)

        if self.mode == "output":
            return self.model(x) if cell is None else self.model(x, cell=cell)

        x = self._apply_preprocessing(x, cell)

        if self.mode == "forward":
            return self.model.forward_cv(x)

        norm_in = getattr(self.model, "norm_in", None)
        if norm_in is not None:
            x = norm_in(x)

        return self._validate_output(self.model.nn(x))


class _GraphMLColvarRepresentation(
    _FrozenModelMixin,
    GraphRepresentation,
):
    """Representation adapter for graph-based mlcolvar models."""

    def __init__(
        self,
        model: nn.Module,
        *,
        out_features: Optional[int],
        freeze: bool,
    ) -> None:
        encoder = getattr(model, "nn", None)

        if not isinstance(encoder, BaseGNN):
            raise TypeError(
                f"{model.__class__.__name__} is not graph-based: "
                "expected `.nn` to be BaseGNN."
            )

        pooling_operation = getattr(
            encoder,
            "pooling_operation",
            None,
        )

        output_kind = (
            "atom"
            if pooling_operation is None
            else "system"
        )

        resolved_out = as_positive_int(
            encoder.out_features if out_features is None else out_features,
            "encoder.out_features" if out_features is None else "out_features",
        )

        GraphRepresentation.__init__(
            self,
            out_features=resolved_out,
            atomic_numbers=to_int_list(
                encoder.atomic_numbers,
                name="encoder.atomic_numbers",
            ),
            cutoff=to_float(
                encoder.cutoff,
                name="encoder.cutoff",
            ),
            output_kind=output_kind,
            buffer=to_float(
                encoder.buffer,
                name="encoder.buffer",
            ),
            long_range_cutoff=to_float(
                encoder.long_range_cutoff,
                name="encoder.long_range_cutoff",
            ),
            full_neighbor_list=True,
            freeze=freeze,
        )

        self.pooling_operation = pooling_operation
        self._init_model(model)

    def _cast_graph(
        self,
        data: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        output: Dict[str, torch.Tensor] = {}

        for key, value in data.items():
            if not torch.is_tensor(value):
                continue

            output[key] = (
                value.to(self._model_reference)
                if value.is_floating_point() or value.is_complex()
                else value.to(self._model_reference.device)
            )

        return output

    def forward(
        self,
        data: Dict[str, torch.Tensor],
        cell: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        data = self._cast_graph(data)

        if cell is not None:
            cell = cell.to(self._model_reference)

        data = self._apply_preprocessing(data, cell)

        if getattr(self.model, "norm_in", None) is not None:
            raise ValueError(
                "Input normalization cannot be applied "
                "directly to graph dictionaries."
            )

        return self._validate_output(
            self.model.nn(data)
        )


def MLColvarRepresentation(
    model: nn.Module,
    *,
    mode: str = "latent",
    out_features: Optional[int] = None,
    freeze: bool = True,
) -> Representation:
    """Wrap a pretrained mlcolvar model as a reusable representation.

    Parameters
    ----------
    model
        Pretrained mlcolvar model.
    mode
        ``"latent"`` uses ``model.nn``, ``"output"`` uses the complete model,
        and ``"forward"`` uses ``model.forward_cv``.
    out_features
        Optional latent dimension override for ``mode="latent"``.
    freeze
        If True, keep the pretrained model frozen and in evaluation mode.
    """
    if not isinstance(model, nn.Module):
        raise TypeError("`model` must be a torch.nn.Module.")

    mode = mode.lower()
    if mode not in {"latent", "output", "forward"}:
        raise ValueError(
            "`mode` must be 'latent', 'output', or 'forward'."
        )

    if isinstance(getattr(model, "nn", None), BaseGNN):
        if mode != "latent":
            raise ValueError(
                "Graph-based pretrained CVs support only mode='latent'."
            )
        return _GraphMLColvarRepresentation(
            model=model,
            out_features=out_features,
            freeze=freeze,
        )

    return _VectorMLColvarRepresentation(
        model=model,
        mode=mode,
        out_features=out_features,
        freeze=freeze,
    )