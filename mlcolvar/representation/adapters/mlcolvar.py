from typing import Dict, Optional

import torch
from torch import nn

from mlcolvar.core import BaseGNN

from .._utils import as_positive_int, module_reference_tensor
from ..base import GraphRepresentation, Representation, VectorRepresentation
from ._utils import to_float, to_int_list

__all__ = ["MLColvarRepresentation"]


def _encoder(model: nn.Module) -> nn.Module:
    encoder = getattr(model, "nn", None)
    if encoder is None:
        raise TypeError(
            f"{model.__class__.__name__} does not expose an `.nn` encoder."
        )
    return encoder


def _out_features(encoder: nn.Module, value: Optional[int]) -> int:
    if value is not None:
        return as_positive_int(value, "out_features")

    for name in ("out_features", "n_out"):
        value = getattr(encoder, name, None)
        if value is not None:
            return as_positive_int(value, name)

    raise ValueError(
        "Cannot infer the representation dimension from "
        f"{encoder.__class__.__name__}; pass `out_features` explicitly."
    )


def _preprocess(model: nn.Module, data, cell=None):
    preprocessing = getattr(model, "preprocessing", None)
    if preprocessing is None:
        return data
    return preprocessing(data) if cell is None else preprocessing(data, cell=cell)


class _VectorMLColvarRepresentation(VectorRepresentation):
    def __init__(
        self,
        model: nn.Module,
        out_features: Optional[int],
        freeze: bool,
    ):
        encoder = _encoder(model)
        in_features = getattr(model, "in_features", None)

        if in_features is None:
            raise TypeError(
                f"{model.__class__.__name__} does not expose vector input features."
            )

        super().__init__(
            in_features=as_positive_int(in_features, "model.in_features"),
            out_features=_out_features(encoder, out_features),
            output_kind="system",
            freeze=freeze,
        )

        self.model = model
        self.register_buffer(
            "_model_reference",
            module_reference_tensor(model),
            persistent=False,
        )

        if freeze:
            self._freeze_module(model)

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
        cell = None if cell is None else cell.to(self._model_reference)

        x = _preprocess(self.model, x, cell)

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


class _GraphMLColvarRepresentation(GraphRepresentation):
    def __init__(
        self,
        model: nn.Module,
        out_features: Optional[int],
        freeze: bool,
    ):
        encoder = _encoder(model)

        super().__init__(
            out_features=_out_features(encoder, out_features),
            atomic_numbers=to_int_list(
                encoder.atomic_numbers,
                name="encoder.atomic_numbers",
            ),
            cutoff=to_float(encoder.cutoff, name="encoder.cutoff"),
            pooling_operation=getattr(encoder, "pooling_operation", None),
            buffer=to_float(encoder.buffer, name="encoder.buffer"),
            long_range_cutoff=to_float(
                encoder.long_range_cutoff,
                name="encoder.long_range_cutoff",
            ),
            full_neighbor_list=True,
            freeze=freeze,
        )

        self.model = model
        self.register_buffer(
            "_model_reference",
            module_reference_tensor(model),
            persistent=False,
        )

        if freeze:
            self._freeze_module(model)

    def forward(
        self,
        data: Dict[str, torch.Tensor],
        cell: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        data = {
            key: (
                value.to(self._model_reference)
                if value.is_floating_point() or value.is_complex()
                else value.to(self._model_reference.device)
            )
            for key, value in data.items()
            if torch.is_tensor(value)
        }

        cell = None if cell is None else cell.to(self._model_reference)
        data = _preprocess(self.model, data, cell)

        if getattr(self.model, "norm_in", None) is not None:
            raise ValueError(
                "Input normalization cannot be applied directly to graph dictionaries."
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
    out_features: Optional[int] = None,
    freeze: bool = True,
) -> Representation:
    """Wrap the latent encoder of a pretrained mlcolvar model."""
    if not isinstance(model, nn.Module):
        raise TypeError("`model` must be a torch.nn.Module.")

    encoder = _encoder(model)

    if isinstance(encoder, BaseGNN):
        return _GraphMLColvarRepresentation(model, out_features, freeze)

    return _VectorMLColvarRepresentation(model, out_features, freeze)