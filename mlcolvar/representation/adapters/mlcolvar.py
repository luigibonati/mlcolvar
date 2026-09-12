from typing import Dict, Optional, Tuple

import torch
from torch import nn

from mlcolvar.core import BaseGNN

from ..base import (
    GraphRepresentation,
    Representation,
    TensorRepresentation,
    as_positive_int,
    module_reference_tensor,
)


__all__ = ["MLColvarRepresentation"]


def _infer_model_output_dimension(model: nn.Module) -> int:
    internal_model = getattr(model, "nn", None)
    if internal_model is None:
        internal_model = model

    out_features = getattr(internal_model, "out_features", None)
    if out_features is not None:
        return as_positive_int(out_features, "out_features")

    n_out = getattr(internal_model, "n_out", None)
    if n_out is not None:
        return as_positive_int(n_out, "n_out")

    raise ValueError(
        "Cannot infer the representation dimension from "
        f"{model.__class__.__name__}; pass `out_features` explicitly."
    )


def _is_graph_model(model: nn.Module) -> bool:
    return isinstance(getattr(model, "nn", None), BaseGNN)


class _FrozenModelMixin:
    def _init_model(self, model: nn.Module) -> None:
        self.model = model
        self.register_buffer(
            "_model_reference",
            module_reference_tensor(model),
            persistent=False,
        )
        self._freeze_module(model)

    def _keep_frozen_modules_in_eval(self) -> None:
        self.model.eval()

    def _apply_preprocessing(self, data, cell=None):
        preprocessing = getattr(self.model, "preprocessing", None)

        if preprocessing is None:
            return data

        if cell is None:
            return preprocessing(data)

        return preprocessing(data, cell=cell)


class _TensorMLColvarRepresentation(
    _FrozenModelMixin,
    TensorRepresentation,
):
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
            raise TypeError(
                f"{model.__class__.__name__} is graph-based."
            )

        if mode == "latent":
            latent = getattr(model, "nn", None)

            if latent is None:
                raise TypeError(
                    f"{model.__class__.__name__} does not expose "
                    "an `.nn` latent encoder."
                )

            resolved_out = (
                _infer_model_output_dimension(model)
                if out_features is None
                else as_positive_int(
                    out_features,
                    "out_features",
                )
            )

        elif mode == "output":
            if out_features is not None:
                raise ValueError(
                    "`out_features` is only valid with mode='latent'."
                )

            resolved_out = as_positive_int(
                model.out_features,
                "model.out_features",
            )

        elif mode == "forward":
            if out_features is not None:
                raise ValueError(
                    "`out_features` is only valid with mode='latent'."
                )

            if not hasattr(model, "forward_cv"):
                raise TypeError(
                    f"{model.__class__.__name__} does not implement "
                    "`forward_cv()`."
                )

            resolved_out = as_positive_int(
                model.out_features,
                "model.out_features",
            )

        else:
            raise ValueError(
                "`mode` must be 'latent', 'output', or 'forward'."
            )

        TensorRepresentation.__init__(
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

    def _cast_input(
        self,
        x: torch.Tensor,
        cell: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        x = x.to(
            device=self._model_reference.device,
            dtype=self._model_reference.dtype,
        )

        if cell is not None:
            cell = cell.to(
                device=self._model_reference.device,
                dtype=self._model_reference.dtype,
            )

        return x, cell

    def forward(
        self,
        x: torch.Tensor,
        cell: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if x.ndim < 2:
            raise ValueError(
                "`x` must include a batch dimension."
            )

        if x.shape[-1] != self.in_features:
            raise ValueError(
                f"Expected {self.in_features} input features, "
                f"found {x.shape[-1]}."
            )

        x, cell = self._cast_input(x, cell)

        if self.mode == "output":
            if cell is None:
                return self.model(x)
            return self.model(x, cell=cell)

        x = self._apply_preprocessing(x, cell)

        if self.mode == "forward":
            return self.model.forward_cv(x)

        norm_in = getattr(self.model, "norm_in", None)
        if norm_in is not None:
            x = norm_in(x)

        output = self.model.nn(x)

        if output.shape[-1] != self.out_features:
            raise ValueError(
                f"Expected {self.out_features} latent features, "
                f"found {output.shape[-1]}."
            )

        return output


class _GraphMLColvarRepresentation(
    _FrozenModelMixin,
    GraphRepresentation,
):
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

        if getattr(encoder, "pooling_operation", None) is None:
            raise ValueError(
                "Graph representation transfer requires a graph-level "
                "encoder with `pooling_operation` enabled."
            )

        resolved_out = (
            as_positive_int(
                encoder.out_features,
                "encoder.out_features",
            )
            if out_features is None
            else as_positive_int(
                out_features,
                "out_features",
            )
        )

        GraphRepresentation.__init__(
            self,
            out_features=resolved_out,
            atomic_numbers=encoder.atomic_numbers.detach().cpu().tolist(),
            cutoff=float(encoder.cutoff.detach().cpu().item()),
            output_kind="system",
            buffer=float(encoder.buffer.detach().cpu().item()),
            long_range_cutoff=float(
                encoder.long_range_cutoff.detach().cpu().item()
            ),
            full_neighbor_list=True,
            freeze=freeze,
        )

        self.pooling_operation = encoder.pooling_operation
        self._init_model(model)

    def _cast_graph(
        self,
        data: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        output: Dict[str, torch.Tensor] = {}

        for key, value in data.items():
            if not torch.is_tensor(value):
                continue

            if value.is_floating_point() or value.is_complex():
                output[key] = value.to(
                    dtype=self._model_reference.dtype,
                    device=self._model_reference.device,
                )
            else:
                output[key] = value.to(
                    device=self._model_reference.device,
                )

        return output

    def forward(
        self,
        data: Dict[str, torch.Tensor],
        cell: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        data = self._cast_graph(data)

        if cell is not None:
            cell = cell.to(
                dtype=self._model_reference.dtype,
                device=self._model_reference.device,
            )

        data = self._apply_preprocessing(data, cell)

        if getattr(self.model, "norm_in", None) is not None:
            raise ValueError(
                "Tensor input normalization cannot be applied "
                "directly to graph dictionaries."
            )

        output = self.model.nn(data)

        if output.shape[-1] != self.out_features:
            raise ValueError(
                f"Expected {self.out_features} latent features, "
                f"found {output.shape[-1]}."
            )

        return output


def MLColvarRepresentation(
    model: nn.Module,
    *,
    mode: str = "latent",
    out_features: Optional[int] = None,
    freeze: bool = True,
) -> Representation:
    """Adapt a pretrained mlcolvar CV to the unified representation API."""

    if not isinstance(model, nn.Module):
        raise TypeError(
            "`model` must be a torch.nn.Module."
        )

    mode = mode.lower()

    if mode not in {"latent", "output", "forward"}:
        raise ValueError(
            "`mode` must be 'latent', 'output', or 'forward'."
        )

    if _is_graph_model(model):
        if mode != "latent":
            raise ValueError(
                "Graph-based pretrained CVs support only mode='latent'."
            )

        return _GraphMLColvarRepresentation(
            model=model,
            out_features=out_features,
            freeze=freeze,
        )

    return _TensorMLColvarRepresentation(
        model=model,
        mode=mode,
        out_features=out_features,
        freeze=freeze,
    )