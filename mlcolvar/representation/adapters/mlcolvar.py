from typing import Optional

import torch
from torch import nn

from mlcolvar.core import BaseGNN

from .._utils import as_positive_int, module_reference_tensor
from ..base import Representation
from ._utils import to_float, to_int_list

__all__ = ["MLColvarRepresentation"]


def _encoder(model: nn.Module) -> nn.Module:
    """Return the latent encoder of an mlcolvar model."""
    encoder = getattr(model, "nn", None)
    if encoder is None:
        raise TypeError(
            f"{model.__class__.__name__} does not expose an `.nn` encoder."
        )
    return encoder


def _input_kind(model: nn.Module, encoder: nn.Module) -> str:
    """Infer whether the model consumes graph or descriptor inputs."""
    if isinstance(encoder, BaseGNN):
        return "graph"
    if all(hasattr(encoder, name) for name in ("atomic_numbers", "cutoff")):
        return "graph"
    if getattr(model, "in_features", None) is not None:
        return "vector"
    raise TypeError(
        f"Cannot infer the input type of {model.__class__.__name__}. "
        "Expected `in_features` for descriptor inputs or graph metadata "
        "(`atomic_numbers`, `cutoff`) on the encoder."
    )


def _out_features(encoder: nn.Module, value: Optional[int]) -> int:
    """Resolve the encoder output dimension."""
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
    """Apply optional model preprocessing."""
    preprocessing = getattr(model, "preprocessing", None)
    if preprocessing is None:
        return data
    return preprocessing(data) if cell is None else preprocessing(data, cell=cell)


class MLColvarRepresentation(Representation):
    """Reusable latent representation from a pretrained mlcolvar model.

    The representation reuses the latent encoder stored in ``model.nn`` while
    excluding task-specific downstream blocks and postprocessing. The input
    type is inferred automatically from the model.

    Parameters
    ----------
    model : torch.nn.Module
        Pretrained mlcolvar model exposing its latent encoder through ``nn``.
    out_features : int, optional
        Latent dimension. If omitted, inferred from ``encoder.out_features``
        or ``encoder.n_out``.
    freeze : bool, default=True
        Whether to freeze the pretrained model.
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        out_features: Optional[int] = None,
        freeze: bool = True,
    ) -> None:
        if not isinstance(model, nn.Module):
            raise TypeError("`model` must be a torch.nn.Module.")

        encoder = _encoder(model)
        input_kind = _input_kind(model, encoder)
        kwargs = {
            "out_features": _out_features(encoder, out_features),
            "input_kind": input_kind,
            "freeze": freeze,
        }

        if input_kind == "vector":
            kwargs.update(
                in_features=as_positive_int(
                    model.in_features,
                    "model.in_features",
                ),
                output_kind="system",
            )
        else:
            kwargs.update(
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
                    getattr(encoder, "buffer", 0.0),
                    name="encoder.buffer",
                ),
                long_range_cutoff=to_float(
                    getattr(encoder, "long_range_cutoff", -1.0),
                    name="encoder.long_range_cutoff",
                ),
                full_neighbor_list=True,
            )

        super().__init__(**kwargs)
        self.model = model
        self.register_buffer(
            "_model_reference",
            module_reference_tensor(model),
            persistent=False,
        )
        self._freeze_module(model)

    def forward(
        self,
        data,
        cell: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Evaluate the latent representation."""
        cell = None if cell is None else cell.to(self._model_reference)

        if self.input_kind == "graph":
            data = {
                key: (
                    value.to(self._model_reference)
                    if value.is_floating_point() or value.is_complex()
                    else value.to(self._model_reference.device)
                )
                for key, value in data.items()
                if torch.is_tensor(value)
            }
            data = _preprocess(self.model, data, cell)
            if getattr(self.model, "norm_in", None) is not None:
                raise ValueError(
                    "Input normalization cannot be applied directly "
                    "to graph dictionaries."
                )
        else:
            if data.ndim < 2:
                raise ValueError("`data` must include a batch dimension.")
            if data.shape[-1] != self.in_features:
                raise ValueError(
                    f"Expected {self.in_features} input features, "
                    f"found {data.shape[-1]}."
                )

            data = data.to(self._model_reference)
            data = _preprocess(self.model, data, cell)

            norm_in = getattr(self.model, "norm_in", None)
            if norm_in is not None:
                data = norm_in(data)

        output = self.model.nn(data)
        if output.shape[-1] != self.out_features:
            raise ValueError(
                f"Expected {self.out_features} latent features, "
                f"found {output.shape[-1]}."
            )
        return output