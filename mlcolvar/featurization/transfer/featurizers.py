from typing import Any, Dict, Optional, Tuple

import torch
from torch import nn

from mlcolvar.core import BaseGNN

from ._utils import (
    _as_positive_int,
    _get_graph_encoder,
    _infer_model_output_dimension,
    _is_graph_model,
    _module_reference_tensor,
)


__all__ = ["TransferFeaturizer"]


class _BaseTransferFeaturizer(nn.Module):
    """Shared functionality for frozen pretrained-CV featurizers."""

    __constants__ = ["out_features", "freeze"]

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

        self.model = model
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

    def _apply_preprocessing(
        self,
        data: Any,
        cell: Optional[torch.Tensor] = None,
    ) -> Any:
        preprocessing = getattr(
            self.model,
            "preprocessing",
            None,
        )

        if preprocessing is None:
            return data

        if cell is None:
            return preprocessing(data)

        return preprocessing(data, cell=cell)


class _BaseTensorFeaturizer(_BaseTransferFeaturizer):
    """Base implementation for descriptor/FFNN pretrained CVs."""

    def __init__(
        self,
        model: nn.Module,
        out_features: int,
        freeze: bool = True,
    ) -> None:
        if not hasattr(model, "in_features"):
            raise AttributeError(
                f"{model.__class__.__name__} "
                "does not define `in_features`."
            )

        if (
            model.in_features is None
            or isinstance(
                getattr(model, "nn", None),
                BaseGNN,
            )
        ):
            raise TypeError(
                f"{model.__class__.__name__} is graph-based; "
                "use `TransferFeaturizer(..., mode='latent')` "
                "with the graph model."
            )

        self.in_features = _as_positive_int(
            model.in_features,
            "in_features",
        )

        super().__init__(
            model=model,
            out_features=out_features,
            freeze=freeze,
        )

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

    @torch.no_grad()
    def precompute(
        self,
        x: torch.Tensor,
        batch_size: int = 4096,
        device: Optional[str] = None,
        output_device: str = "cpu",
    ) -> torch.Tensor:
        """Precompute frozen tensor features for head-only training."""

        if not self.freeze:
            raise RuntimeError(
                "`precompute` requires freeze=True."
            )

        if batch_size <= 0:
            raise ValueError(
                "`batch_size` must be a positive integer."
            )

        if len(x) == 0:
            raise ValueError(
                "Cannot precompute features from an empty tensor."
            )

        original_device = self._model_reference.device
        was_training = self.training
        target_device = torch.device(
            device or original_device
        )

        try:
            self.to(target_device).eval()

            outputs = []
            for start in range(0, len(x), batch_size):
                batch = x[
                    start : start + batch_size
                ].to(target_device)

                outputs.append(
                    self(batch).to(output_device)
                )

            return torch.cat(outputs, dim=0)

        finally:
            self.to(original_device)
            self.train(was_training)


class _BaseGraphFeaturizer(_BaseTransferFeaturizer):
    """Base implementation for GNN pretrained CVs."""

    def __init__(
        self,
        model: nn.Module,
        out_features: int,
        freeze: bool = True,
    ) -> None:
        encoder = _get_graph_encoder(model)

        if encoder.pooling_operation is None:
            raise ValueError(
                "The pretrained GNN must use graph-level pooling "
                "before transfer. Set `pooling_operation` to "
                "'mean' or 'sum'."
            )

        self.in_features = None
        self.pooling_operation = encoder.pooling_operation

        super().__init__(
            model=model,
            out_features=out_features,
            freeze=freeze,
        )

        for name in (
            "cutoff",
            "buffer",
            "long_range_cutoff",
            "atomic_numbers",
        ):
            self.register_buffer(
                name,
                getattr(encoder, name).detach().clone(),
            )

    def _cast_graph(
        self,
        data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Move graph tensors to the pretrained model device and precision."""

        output: Dict[str, Any] = {}

        for key, value in data.items():
            if not torch.is_tensor(value):
                output[key] = value
            elif (
                value.is_floating_point()
                or value.is_complex()
            ):
                output[key] = value.to(
                    dtype=self._model_reference.dtype,
                    device=self._model_reference.device,
                )
            else:
                output[key] = value.to(
                    device=self._model_reference.device,
                )

        return output


class _OutputFeaturizer(_BaseTensorFeaturizer):
    """Use the complete output of a tensor-based pretrained CV."""

    def __init__(
        self,
        model: nn.Module,
        freeze: bool = True,
    ) -> None:
        if not hasattr(model, "out_features"):
            raise AttributeError(
                f"{model.__class__.__name__} "
                "does not define `out_features`."
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


class _ForwardFeaturizer(_BaseTensorFeaturizer):
    """Use preprocessing followed by ``forward_cv``."""

    def __init__(
        self,
        model: nn.Module,
        freeze: bool = True,
    ) -> None:
        if not hasattr(model, "forward_cv"):
            raise TypeError(
                f"{model.__class__.__name__} "
                "does not implement `forward_cv()`."
            )

        if not hasattr(model, "out_features"):
            raise AttributeError(
                f"{model.__class__.__name__} "
                "does not define `out_features`."
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


class _TensorLatentFeaturizer(_BaseTensorFeaturizer):
    """Use the latent encoder representation of a tensor-based CV."""

    def __init__(
        self,
        model: nn.Module,
        out_features: Optional[int] = None,
        freeze: bool = True,
    ) -> None:
        internal_model = getattr(model, "nn", None)

        if internal_model is None:
            raise TypeError(
                f"{model.__class__.__name__} "
                "does not define an `nn` block."
            )

        if isinstance(internal_model, BaseGNN):
            raise TypeError(
                "The pretrained model is graph-based."
            )

        if out_features is None:
            out_features = _infer_model_output_dimension(
                model
            )

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
        x = self._apply_preprocessing(x, cell)

        norm_in = getattr(
            self.model,
            "norm_in",
            None,
        )

        if norm_in is not None:
            x = norm_in(x)

        output = self.model.nn(x)

        if output.shape[-1] != self.out_features:
            raise ValueError(
                f"Expected {self.out_features} latent features, "
                f"found {output.shape[-1]}."
            )

        return output


class _GraphLatentFeaturizer(_BaseGraphFeaturizer):
    """Use the latent encoder representation of a graph-based CV."""

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
        data: Dict[str, Any],
        cell: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        data = self._cast_graph(data)

        if cell is not None:
            cell = cell.to(
                dtype=self._model_reference.dtype,
                device=self._model_reference.device,
            )

        data = self._apply_preprocessing(
            data,
            cell,
        )

        if getattr(
            self.model,
            "norm_in",
            None,
        ) is not None:
            raise ValueError(
                "Input normalization is tensor-based and cannot be "
                "applied directly to graph dictionaries. Disable "
                "`norm_in` for the GNN pretrained model."
            )

        output = self.model.nn(data)

        if (
            not torch.jit.is_tracing()
            and output.shape[-1] != self.out_features
        ):
            raise ValueError(
                f"Expected {self.out_features} graph latent features, "
                f"found {output.shape[-1]}."
            )

        return output


def TransferFeaturizer(
    model: nn.Module,
    mode: str = "latent",
    out_features: Optional[int] = None,
    freeze: bool = True,
) -> nn.Module:
    """Create a featurizer from a pretrained mlcolvar CV.

    Parameters
    ----------
    model
        Pretrained mlcolvar CV.
    mode
        Representation to reuse:

        - ``"latent"``: encoder representation
          (tensor or graph models);
        - ``"output"``: complete model output
          (tensor models only);
        - ``"forward"``: preprocessing + ``forward_cv``
          (tensor models only).

    out_features
        Optional latent dimension override.
        Only used with ``mode="latent"``.
    freeze
        Freeze pretrained model parameters while preserving gradients
        with respect to the input coordinates/features.
    """

    if not isinstance(model, nn.Module):
        raise TypeError(
            "`model` must be an instance of torch.nn.Module. "
            f"Found {type(model)}."
        )

    mode = mode.lower()

    if mode not in (
        "latent",
        "output",
        "forward",
    ):
        raise ValueError(
            "`mode` must be 'latent', 'output', or 'forward'. "
            f"Found {mode!r}."
        )

    if _is_graph_model(model):
        if mode != "latent":
            raise ValueError(
                "Graph-based pretrained CVs currently support "
                "only `mode='latent'`."
            )

        return _GraphLatentFeaturizer(
            model=model,
            out_features=out_features,
            freeze=freeze,
        )

    if mode == "latent":
        return _TensorLatentFeaturizer(
            model=model,
            out_features=out_features,
            freeze=freeze,
        )

    if out_features is not None:
        raise ValueError(
            "`out_features` can only be specified "
            "with `mode='latent'`."
        )

    if mode == "output":
        return _OutputFeaturizer(
            model=model,
            freeze=freeze,
        )

    return _ForwardFeaturizer(
        model=model,
        freeze=freeze,
    )