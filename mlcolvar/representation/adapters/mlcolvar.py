from typing import Dict, Optional

import torch
from torch import nn

from mlcolvar.core import BaseGNN

from .._utils import as_positive_int, module_reference_tensor
from ..base import GraphRepresentation, Representation, VectorRepresentation
from ._utils import to_float, to_int_list

__all__ = ["MLColvarRepresentation"]


def _latent_encoder(model: nn.Module) -> nn.Module:
    """Return the latent encoder exposed as ``model.nn``."""
    encoder = getattr(model, "nn", None)
    if encoder is None:
        raise TypeError(
            f"{model.__class__.__name__} does not expose an `.nn` latent encoder."
        )
    return encoder


def _infer_output_dimension(encoder: nn.Module) -> int:
    """Infer the latent representation dimension from an encoder."""
    for name in ("out_features", "n_out"):
        value = getattr(encoder, name, None)
        if value is not None:
            return as_positive_int(value, name)

    raise ValueError(
        "Cannot infer the representation dimension from "
        f"{encoder.__class__.__name__}; pass `out_features` explicitly."
    )


class _FrozenModelMixin:
    """Utilities shared by pretrained mlcolvar representations."""

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
        return (
            preprocessing(data)
            if cell is None
            else preprocessing(data, cell=cell)
        )

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
    """Adapt the latent encoder of a vector-based mlcolvar model."""

    def __init__(
        self,
        model: nn.Module,
        *,
        out_features: Optional[int],
        freeze: bool,
    ) -> None:
        encoder = _latent_encoder(model)

        if isinstance(encoder, BaseGNN):
            raise TypeError(f"{model.__class__.__name__} is graph-based.")

        if getattr(model, "in_features", None) is None:
            raise TypeError(
                f"{model.__class__.__name__} does not expose vector input features."
            )

        resolved_out = (
            _infer_output_dimension(encoder)
            if out_features is None
            else as_positive_int(out_features, "out_features")
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
        self._init_model(model)

    def forward(
        self,
        x: torch.Tensor,
        cell: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Evaluate the pretrained latent representation."""
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

        x = self._apply_preprocessing(x, cell)

        norm_in = getattr(self.model, "norm_in", None)
        if norm_in is not None:
            x = norm_in(x)

        return self._validate_output(self.model.nn(x))


class _GraphMLColvarRepresentation(
    _FrozenModelMixin,
    GraphRepresentation,
):
    """Adapt the latent encoder of a graph-based mlcolvar model."""

    def __init__(
        self,
        model: nn.Module,
        *,
        out_features: Optional[int],
        freeze: bool,
    ) -> None:
        encoder = _latent_encoder(model)

        if not isinstance(encoder, BaseGNN):
            raise TypeError(
                f"{model.__class__.__name__} is not graph-based: "
                "expected `.nn` to be BaseGNN."
            )

        resolved_out = (
            _infer_output_dimension(encoder)
            if out_features is None
            else as_positive_int(out_features, "out_features")
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
            pooling_operation=getattr(
                encoder,
                "pooling_operation",
                None,
            ),
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
        self._init_model(model)

    def _cast_graph(
        self,
        data: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Move graph tensors to the wrapped model device and dtype."""
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
        """Evaluate the pretrained graph representation."""
        data = self._cast_graph(data)

        if cell is not None:
            cell = cell.to(self._model_reference)

        data = self._apply_preprocessing(data, cell)

        if getattr(self.model, "norm_in", None) is not None:
            raise ValueError(
                "Input normalization cannot be applied "
                "directly to graph dictionaries."
            )

        return self._validate_output(self.model.nn(data))


def MLColvarRepresentation(
    model: nn.Module,
    *,
    out_features: Optional[int] = None,
    freeze: bool = True,
) -> Representation:
    """Wrap the latent encoder of a pretrained mlcolvar model.

    The representation includes any preprocessing and input normalization
    applied before ``model.nn``, while excluding downstream task-specific
    blocks and postprocessing.

    Parameters
    ----------
    model : torch.nn.Module
        Pretrained mlcolvar model exposing its latent encoder as ``model.nn``.
    out_features : int, optional
        Latent representation dimension. If not provided, it is inferred from
        the encoder.
    freeze : bool, default=True
        If True, freeze the pretrained model parameters and keep the wrapped
        model in evaluation mode.

    Returns
    -------
    Representation
        Vector or graph representation matching the pretrained model.

    Raises
    ------
    TypeError
        If ``model`` is not a module or does not expose a valid latent encoder.
    ValueError
        If the latent representation dimension cannot be inferred.
    """
    if not isinstance(model, nn.Module):
        raise TypeError("`model` must be a torch.nn.Module.")

    encoder = _latent_encoder(model)

    if isinstance(encoder, BaseGNN):
        return _GraphMLColvarRepresentation(
            model=model,
            out_features=out_features,
            freeze=freeze,
        )

    return _VectorMLColvarRepresentation(
        model=model,
        out_features=out_features,
        freeze=freeze,
    )