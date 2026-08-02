from typing import Any, Dict, Optional, Tuple

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

        internal_model = getattr(
            model,
            "nn",
            None,
        )

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

    def train(
        self,
        mode: bool = True,
    ):
        super().train(mode)

        if self.freeze:
            self.model.eval()

        return self

    def _cast_input(
        self,
        x: torch.Tensor,
        cell: Optional[torch.Tensor] = None,
    ) -> Tuple[
        torch.Tensor,
        Optional[torch.Tensor],
    ]:
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
        preprocessing = getattr(
            self.model,
            "preprocessing",
            None,
        )

        if preprocessing is None:
            return x

        if cell is None:
            return preprocessing(x)

        return preprocessing(
            x,
            cell=cell,
        )

    @torch.no_grad()
    def precompute(
        self,
        x: torch.Tensor,
        batch_size: int = 4096,
        device: Optional[str] = None,
        output_device: str = "cpu",
    ) -> torch.Tensor:
        """Precompute frozen features once for head-only training."""

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

        device = torch.device(
            device or self._model_reference.device
        )

        self.to(device).eval()

        outputs = []

        for start in range(
            0,
            len(x),
            batch_size,
        ):
            stop = min(
                start + batch_size,
                len(x),
            )

            batch = x[start:stop].to(device)
            features = self(batch)

            outputs.append(
                features.to(output_device)
            )

        return torch.cat(
            outputs,
            dim=0,
        )


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
        self.pooling_operation = (
            encoder.pooling_operation
        )

        # A downstream graph-level task requires one feature vector
        # per graph. Node-level outputs would not match graph labels.
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
            encoder.long_range_cutoff
            .detach()
            .clone(),
        )
        self.register_buffer(
            "atomic_numbers",
            encoder.atomic_numbers
            .detach()
            .clone(),
        )
        self.register_buffer(
            "_model_reference",
            _module_reference_tensor(model),
            persistent=False,
        )

        if self.freeze:
            self.model.requires_grad_(False)
            self.model.eval()

    def train(
        self,
        mode: bool = True,
    ):
        super().train(mode)

        if self.freeze:
            self.model.eval()

        return self

    def _cast_graph(
        self,
        data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Move graph tensors to the pretrained model device/precision.

        PyG graph objects may contain non-tensor metadata such as
        ``num_nodes``. Such values are preserved unchanged.

        Floating-point and complex tensors are converted to the model
        dtype and device. Integer and boolean tensors are moved to the
        model device without changing their dtype.
        """

        output: Dict[str, Any] = {}

        for key, value in data.items():
            # PyG Data/Batch can contain Python metadata:
            # num_nodes, names, identifiers, and similar fields.
            if not torch.is_tensor(value):
                output[key] = value
                continue

            if (
                value.is_floating_point()
                or value.is_complex()
            ):
                output[key] = value.to(
                    dtype=self._model_reference.dtype,
                    device=self._model_reference.device,
                )
            else:
                # Preserve long/bool dtype for edge_index, batch,
                # ptr, masks, atom types, and related tensors.
                output[key] = value.to(
                    device=self._model_reference.device,
                )

        return output

    def _apply_preprocessing(
        self,
        data: Dict[str, Any],
        cell: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        preprocessing = getattr(
            self.model,
            "preprocessing",
            None,
        )

        if preprocessing is None:
            return data

        if cell is None:
            return preprocessing(data)

        return preprocessing(
            data,
            cell=cell,
        )


class CVOutputFeaturizer(
    _BaseTensorCVFeaturizer
):
    """Use the complete output of an FFNN-based pretrained CV."""

    def __init__(
        self,
        model: nn.Module,
        freeze: bool = True,
    ) -> None:
        if not hasattr(
            model,
            "out_features",
        ):
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
        x, cell = self._cast_input(
            x,
            cell,
        )

        if cell is None:
            return self.model(x)

        return self.model(
            x,
            cell=cell,
        )


class CVForwardFeaturizer(
    _BaseTensorCVFeaturizer
):
    """Use preprocessing + ``forward_cv`` of an FFNN-based CV."""

    def __init__(
        self,
        model: nn.Module,
        freeze: bool = True,
    ) -> None:
        if not hasattr(
            model,
            "forward_cv",
        ):
            raise TypeError(
                f"{model.__class__.__name__} "
                "does not implement `forward_cv()`."
            )

        if not hasattr(
            model,
            "out_features",
        ):
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
        x, cell = self._cast_input(
            x,
            cell,
        )

        x = self._apply_preprocessing(
            x,
            cell,
        )

        return self.model.forward_cv(x)


class CVLatentFeaturizer(
    _BaseTensorCVFeaturizer
):
    """Use the latent encoder representation of an FFNN-based CV.

    This implementation directly evaluates:

    preprocessing -> norm_in -> nn

    rather than calling ``model.forward_nn``. This also works when the
    pretrained SelfTICA model was constructed from an external
    FeedForward module.
    """

    def __init__(
        self,
        model: nn.Module,
        out_features: Optional[int] = None,
        freeze: bool = True,
    ) -> None:
        internal_model = getattr(
            model,
            "nn",
            None,
        )

        if internal_model is None:
            raise TypeError(
                f"{model.__class__.__name__} "
                "does not define an `nn` block."
            )

        if isinstance(
            internal_model,
            BaseGNN,
        ):
            raise TypeError(
                "The pretrained model is graph-based. Use "
                "`CVGraphLatentFeaturizer`."
            )

        if out_features is None:
            out_features = (
                _infer_model_output_dimension(
                    model
                )
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

        x, cell = self._cast_input(
            x,
            cell,
        )

        x = self._apply_preprocessing(
            x,
            cell,
        )

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


class CVGraphLatentFeaturizer(
    _BaseGraphCVFeaturizer
):
    """Use the latent encoder representation of a GNN-based CV.

    The wrapper directly evaluates the pretrained CV's ``nn`` block.
    This avoids relying on SelfTICA.forward_nn, whose current
    override-model path does not return the encoder output.
    """

    def __init__(
        self,
        model: nn.Module,
        out_features: Optional[int] = None,
        freeze: bool = True,
    ) -> None:
        encoder = _get_graph_encoder(
            model
        )

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

        data = self._apply_preprocessing(
            data,
            cell,
        )

        norm_in = getattr(
            self.model,
            "norm_in",
            None,
        )

        if norm_in is not None:
            raise ValueError(
                "Input Normalization is tensor-based and cannot be "
                "applied directly to graph dictionaries. Disable "
                "`norm_in` for the GNN SelfTICA model."
            )

        output = self.model.nn(data)

        if (
            not torch.jit.is_tracing()
            and output.shape[-1]
            != self.out_features
        ):
            raise ValueError(
                f"Expected {self.out_features} graph latent features, "
                f"found {output.shape[-1]}."
            )

        return output