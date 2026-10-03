import torch
from torch import nn

from mlcolvar.core import BaseGNN

from .._utils import as_float, as_positive_int, module_reference_tensor
from ..base import Representation

__all__ = ["MLColvarRepresentation"]


class MLColvarRepresentation(Representation):
    """Reusable latent representation from a pretrained mlcolvar model."""

    def __init__(
        self,
        model: nn.Module,
        *,
        freeze: bool = True,
    ) -> None:
        if not isinstance(model, nn.Module):
            raise TypeError("`model` must be a torch.nn.Module.")
        encoder = getattr(model, "nn", None)
        if encoder is None:
            raise TypeError(
                f"{model.__class__.__name__} does not expose an `.nn` encoder."
            )

        out_features = as_positive_int(
            encoder.out_features, "encoder.out_features"
        )
        if isinstance(encoder, BaseGNN):
            super().__init__(
                out_features=out_features,
                input_kind="graph",
                atomic_numbers=encoder.atomic_numbers,
                cutoff=as_float(encoder.cutoff, "encoder.cutoff"),
                pooling_operation=encoder.pooling_operation,
                buffer=as_float(encoder.buffer, "encoder.buffer"),
                long_range_cutoff=as_float(
                    encoder.long_range_cutoff,
                    "encoder.long_range_cutoff",
                ),
                freeze=freeze,
            )
        else:
            super().__init__(
                out_features=out_features,
                input_kind="vector",
                output_kind="system",
                in_features=as_positive_int(
                    model.in_features, "model.in_features"
                ),
                freeze=freeze,
            )

        self.encoder = encoder
        self.preprocessing = getattr(model, "preprocessing", None)
        self.norm_in = getattr(model, "norm_in", None)
        self.register_buffer(
            "_model_reference",
            module_reference_tensor(encoder),
            persistent=False,
        )
        self._freeze_module(self.encoder)
        if isinstance(self.preprocessing, nn.Module):
            self._freeze_module(self.preprocessing)
        if isinstance(self.norm_in, nn.Module):
            self._freeze_module(self.norm_in)

    def forward(
        self,
        data,
        cell: torch.Tensor | None = None,
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
            if self.preprocessing is not None:
                data = (
                    self.preprocessing(data)
                    if cell is None
                    else self.preprocessing(data, cell=cell)
                )
            if self.norm_in is not None:
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
            if self.preprocessing is not None:
                data = (
                    self.preprocessing(data)
                    if cell is None
                    else self.preprocessing(data, cell=cell)
                )
            if self.norm_in is not None:
                data = self.norm_in(data)

        output = self.encoder(data)
        if output.shape[-1] != self.out_features:
            raise ValueError(
                f"Expected {self.out_features} latent features, "
                f"found {output.shape[-1]}."
            )
        return output