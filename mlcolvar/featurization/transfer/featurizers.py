from typing import Dict, Optional, Tuple

import torch
from torch import nn

from mlcolvar.core import BaseGNN

from ._utils import (
    _as_positive_int,
    _get_graph_encoder,
    _infer_model_output_dimension,
    _module_reference_tensor,
)


__all__ = [
    "CVOutputFeaturizer",
    "CVForwardFeaturizer",
    "CVLatentFeaturizer",
    "CVGraphLatentFeaturizer",
]


class _BaseTensorCVFeaturizer(nn.Module):
    """Base wrapper for descriptor/FFNN-based pretrained mlcolvar CVs."""

    __constants__ = [
        "in_features",
        "out_features",
        "freeze",
    ]

    def __init__(
        self,
        model: nn.Module,
        out_features: int,
        freeze: bool = True,
    ) -> None:
        super().__init__()

        if not isinstance(model, nn.Module):
            raise TypeError(
                "`model` must be an instance of torch.nn.Module. "
                f"Found {type(model)}."
            )

        if not hasattr(model, "in_features"):
            raise AttributeError(
                f"{model.__class__.__name__} does not define `in_features`."
            )

        if model.in_features is None:
            raise TypeError(
                f"{model.__class__.__name__} is graph-based. Use "
                "`CVGraphLatentFeaturizer` and `CVGraphReadoutModel`."
            )

        internal_model = getattr(model, "nn", None)
        if isinstance(internal_model, BaseGNN):
            raise TypeError(
                f"{model.__class__.__name__} contains a BaseGNN. Use "
                "`CVGraphLatentFeaturizer` and `CVGraphReadoutModel`."
            )

        self.model = model
        self.in_features = _as_positive_int(
            model.in_features,
            "in_features",
        )
        self.out_features = _as_positive_int(
            out_features,
            "out_features",
        )
        self.freeze = bool(freeze)

        self.register_buffer(
            "_model_reference",
            _module_reference_tensor(model),
            persistent=False,
        )

        if self.freeze:
            self.model.requires_grad_(False)
            self.model.eval()

    def train(self, mode: bool = True):
        super().train(mode)

        if self.freeze:
            self.model.eval()

        return self

    def _cast_input(
        self,
        x: torch.Tensor,
        cell: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        x = x.to(
            dtype=self._model_reference.dtype,
            device=self._model_reference.device,
        )

        if cell is not None:
            cell = cell.to(
                dtype=self._model_reference.dtype,
                device=self._model_reference.device,
            )

        return x, cell

    def _apply_preprocessing(
        self,
        x: torch.Tensor,
        cell: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        preprocessing = getattr(self.model, "preprocessing", None)

        if preprocessing is None:
            return x

        if cell is None:
            return preprocessing(x)

        return preprocessing(x, cell=cell)


class _BaseGraphCVFeaturizer(nn.Module):
    """Base wrapper for GNN-based pretrained mlcolvar CVs."""

    __constants__ = [
        "out_features",
        "freeze",
    ]

    def __init__(
        self,
        model: nn.Module,
        out_features: int,
        freeze: bool = True,
    ) -> None:
        super().__init__()

        if not isinstance(model, nn.Module):
            raise TypeError(
                "`model` must be an instance of torch.nn.Module. "
                f"Found {type(model)}."
            )

        encoder = _get_graph_encoder(model)

        self.model = model
        self.in_features = None
        self.out_features = _as_positive_int(
            out_features,
            "out_features",
        )
        self.freeze = bool(freeze)
        self.pooling_operation = encoder.pooling_operation

        # A downstream graph-level committor requires one feature vector
        # per graph. Node-level encoder outputs would not match graph labels.
        if self.pooling_operation is None:
            raise ValueError(
                "The pretrained GNN must use graph-level pooling before "
                "transfer to a graph-level readout. Set its "
                "`pooling_operation` to 'mean' or 'sum'."
            )

        self.register_buffer(
            "cutoff",
            encoder.cutoff.detach().clone(),
        )
        self.register_buffer(
            "buffer",
            encoder.buffer.detach().clone(),
        )
        self.register_buffer(
            "long_range_cutoff",
            encoder.long_range_cutoff.detach().clone(),
        )
        self.register_buffer(
            "atomic_numbers",
            encoder.atomic_numbers.detach().clone(),
        )
        self.register_buffer(
            "_model_reference",
            _module_reference_tensor(model),
            persistent=False,
        )

        if self.freeze:
            self.model.requires_grad_(False)
            self.model.eval()

    def train(self, mode: bool = True):
        super().train(mode)

        if self.freeze:
            self.model.eval()

        return self

    def _cast_graph(
        self,
        data: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Move graph tensors to the pretrained model device/precision."""
        output: Dict[str, torch.Tensor] = {}

        for key, value in data.items():
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

    def _apply_preprocessing(
        self,
        data: Dict[str, torch.Tensor],
        cell: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        preprocessing = getattr(self.model, "preprocessing", None)

        if preprocessing is None:
            return data

        if cell is None:
            return preprocessing(data)

        return preprocessing(data, cell=cell)


class CVOutputFeaturizer(_BaseTensorCVFeaturizer):
    """Use the complete output of an FFNN-based pretrained CV."""

    def __init__(
        self,
        model: nn.Module,
        freeze: bool = True,
    ) -> None:
        if not hasattr(model, "out_features"):
            raise AttributeError(
                f"{model.__class__.__name__} does not define `out_features`."
            )

        super().__init__(
            model=model,
            out_features=_as_positive_int(
                model.out_features,
                "out_features",
            ),
            freeze=freeze,
        )

    def forward(
        self,
        x: torch.Tensor,
        cell: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x, cell = self._cast_input(x, cell)

        if cell is None:
            return self.model(x)

        return self.model(x, cell=cell)


class CVForwardFeaturizer(_BaseTensorCVFeaturizer):
    """Use preprocessing + ``forward_cv`` of an FFNN-based CV."""

    def __init__(
        self,
        model: nn.Module,
        freeze: bool = True,
    ) -> None:
        if not hasattr(model, "forward_cv"):
            raise TypeError(
                f"{model.__class__.__name__} does not implement "
                "`forward_cv()`."
            )

        if not hasattr(model, "out_features"):
            raise AttributeError(
                f"{model.__class__.__name__} does not define `out_features`."
            )

        super().__init__(
            model=model,
            out_features=_as_positive_int(
                model.out_features,
                "out_features",
            ),
            freeze=freeze,
        )

    def forward(
        self,
        x: torch.Tensor,
        cell: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x, cell = self._cast_input(x, cell)
        x = self._apply_preprocessing(x, cell)
        return self.model.forward_cv(x)


class CVLatentFeaturizer(_BaseTensorCVFeaturizer):
    """Use the latent encoder representation of an FFNN-based CV.

    This implementation directly evaluates ``preprocessing -> norm_in -> nn``
    rather than calling ``model.forward_nn``. This also works when the
    pretrained SelfTICA was constructed from an external FeedForward module.
    """

    def __init__(
        self,
        model: nn.Module,
        out_features: Optional[int] = None,
        freeze: bool = True,
    ) -> None:
        internal_model = getattr(model, "nn", None)

        if internal_model is None:
            raise TypeError(
                f"{model.__class__.__name__} does not define an `nn` block."
            )

        if isinstance(internal_model, BaseGNN):
            raise TypeError(
                "The pretrained model is graph-based. Use "
                "`CVGraphLatentFeaturizer`."
            )

        if out_features is None:
            out_features = _infer_model_output_dimension(model)

        super().__init__(
            model=model,
            out_features=out_features,
            freeze=freeze,
        )

    def forward(
        self,
        x: torch.Tensor,
        cell: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x, cell = self._cast_input(x, cell)
        x = self._apply_preprocessing(x, cell)

        norm_in = getattr(self.model, "norm_in", None)
        if norm_in is not None:
            x = norm_in(x)

        return self.model.nn(x)


class CVGraphLatentFeaturizer(_BaseGraphCVFeaturizer):
    """Use the latent encoder representation of a GNN-based CV.

    The wrapper directly evaluates the pretrained CV's ``nn`` block. This
    avoids relying on SelfTICA.forward_nn, whose current override-model path
    does not return the encoder output.
    """

    def __init__(
        self,
        model: nn.Module,
        out_features: Optional[int] = None,
        freeze: bool = True,
    ) -> None:
        encoder = _get_graph_encoder(model)

        if out_features is None:
            out_features = _as_positive_int(
                encoder.out_features,
                "encoder.out_features",
            )

        super().__init__(
            model=model,
            out_features=out_features,
            freeze=freeze,
        )

    def forward(
        self,
        data: Dict[str, torch.Tensor],
        cell: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        data = self._cast_graph(data)
        data = self._apply_preprocessing(data, cell)

        norm_in = getattr(self.model, "norm_in", None)
        if norm_in is not None:
            raise ValueError(
                "Input Normalization is tensor-based and cannot be applied "
                "directly to graph dictionaries. Disable `norm_in` for the "
                "GNN SelfTICA model."
            )

        return self.model.nn(data)
