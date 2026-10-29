
import torch
from torch import nn

from mlcolvar.core import BaseGNN

from ._utils import as_float, as_positive_int
from .base import Representation

__all__ = ["MLColvarRepresentation"]


class MLColvarRepresentation(Representation):
    """Extract reusable latent features from a pretrained mlcolvar model.

    Reuses the encoder, preprocessing, and input normalization while
    excluding the original task-specific output layers.

    Parameters
    ----------
    model : nn.Module
        Pretrained model exposing an `.nn` encoder.
    freeze : bool, default=True
        Whether to freeze the pretrained modules.
    """
    def __init__(self, model: nn.Module, *, freeze: bool = True) -> None:
        if not isinstance(model, nn.Module):
            raise TypeError("`model` must be a torch.nn.Module.")

        encoder = getattr(model, "nn", None)
        if encoder is None:
            raise TypeError(f"{type(model).__name__} does not expose an `.nn` encoder.")

        is_graph = isinstance(encoder, BaseGNN)

        if is_graph:
            kwargs = dict(
                atomic_numbers=encoder.atomic_numbers,
                cutoff=as_float(encoder.cutoff, "encoder.cutoff"),
                pooling_operation=encoder.pooling_operation,
                buffer=as_float(encoder.buffer, "encoder.buffer"),
                long_range_cutoff=as_float(
                    encoder.long_range_cutoff, "encoder.long_range_cutoff"
                ),
            )
        else:
            kwargs = dict(
                in_features=as_positive_int(model.in_features, "model.in_features")
            )

        super().__init__(
            out_features=as_positive_int(encoder.out_features, "encoder.out_features"),
            input_kind="graph" if is_graph else "vector",
            output_kind=(
                "atom" if is_graph and encoder.pooling_operation is None else "system"
            ),
            freeze=freeze,
            **kwargs,
        )

        self.encoder = encoder
        self.preprocessing = getattr(model, "preprocessing", None)
        self.norm_in = getattr(model, "norm_in", None)
        
        if self.input_kind == "graph" and self.norm_in is not None:
            raise ValueError("Input normalization is not supported for graph inputs.")

        for module in (self.encoder, self.preprocessing, self.norm_in):
            if isinstance(module, nn.Module):
                self._freeze_module(module)

    def forward(
        self,
        data,
        cell: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Evaluate the pretrained latent representation."""
        if self.preprocessing is not None:
            data = (
                self.preprocessing(data, cell=cell)
                if cell is not None
                else self.preprocessing(data)
            )

        if self.norm_in is not None:
            data = self.norm_in(data)

        return self.encoder(data)
