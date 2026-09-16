from typing import Any, Dict, Optional, Sequence

import torch
from torch import nn

from mlcolvar.core import BaseGNN, FeedForward

from .base import GraphRepresentation, Representation, module_reference_tensor


__all__ = [
    "TaskHead",
    "RepresentationModel",
]


class TaskHead(FeedForward):
    """Small trainable MLP applied on top of a reusable representation."""

    def __init__(
        self,
        in_features: int,
        n_out: int = 1,
        hidden_layers: Sequence[int] = (32, 32),
        options: Optional[Dict[str, Any]] = None,
    ) -> None:
        in_features = int(in_features)
        n_out = int(n_out)
        hidden_layers = tuple(
            int(size) for size in hidden_layers
        )

        if in_features <= 0:
            raise ValueError(
                "`in_features` must be positive."
            )
        if n_out <= 0:
            raise ValueError(
                "`n_out` must be positive."
            )
        if any(size <= 0 for size in hidden_layers):
            raise ValueError(
                "`hidden_layers` must contain positive integers."
            )

        super().__init__(
            layers=[
                in_features,
                *hidden_layers,
                n_out,
            ],
            **(
                {}
                if options is None
                else dict(options)
            ),
        )


class _RepresentationPipelineMixin:
    """Shared representation → head pipeline."""

    def _init_pipeline(
        self,
        *,
        representation: Representation,
        head: nn.Module,
    ) -> None:
        self.representation = representation
        self.head = head

        self.register_buffer(
            "_head_reference",
            module_reference_tensor(head),
            persistent=False,
        )

    @torch.jit.unused
    def cache(
        self,
        dataset,
        *,
        jacobian: bool = False,
        **kwargs,
    ):
        """Precompute representation features and optional Jacobians."""
        from .cache import precompute_representation_cache

        return precompute_representation_cache(
            self.representation,
            dataset,
            compute_jacobian=jacobian,
            **kwargs,
        )


class _TensorRepresentationModel(
    _RepresentationPipelineMixin,
    nn.Module,
):
    """Complete model for tensor-input representations."""

    def __init__(
        self,
        *,
        representation: Representation,
        head: nn.Module,
    ) -> None:
        nn.Module.__init__(self)

        self._init_pipeline(
            representation=representation,
            head=head,
        )

        self.in_features = int(
            representation.in_features
        )
        self.out_features = int(
            head.out_features
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

        features = self.representation(
            x,
            cell=cell,
        )

        features = features.to(
            device=self._head_reference.device,
            dtype=self._head_reference.dtype,
        )

        return self.head(features)


class _GraphRepresentationModel(
    _RepresentationPipelineMixin,
    BaseGNN,
):
    """Complete model for graph-input representations."""

    def __init__(
        self,
        *,
        representation: GraphRepresentation,
        head: nn.Module,
    ) -> None:
        BaseGNN.__init__(
            self,
            n_out=int(head.out_features),
            dataset_for_initialization=None,
            pooling_operation=None,
            cutoff=float(
                representation.cutoff
                .detach()
                .cpu()
                .item()
            ),
            buffer=float(
                representation.buffer
                .detach()
                .cpu()
                .item()
            ),
            long_range_cutoff=float(
                representation.long_range_cutoff
                .detach()
                .cpu()
                .item()
            ),
            atomic_numbers=(
                representation.atomic_numbers
                .detach()
                .cpu()
                .tolist()
            ),
        )

        # Representation adapters construct their own neighborhood features.
        self._modules.pop(
            "_radial_embedding",
            None,
        )

        self._init_pipeline(
            representation=representation,
            head=head,
        )

    def forward(
        self,
        data: Dict[str, torch.Tensor],
        cell: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        features = self.representation(
            data,
            cell=cell,
        )

        features = features.to(
            device=self._head_reference.device,
            dtype=self._head_reference.dtype,
        )

        return self.head(features)


def RepresentationModel(
    representation: Representation,
    *,
    head: Optional[nn.Module] = None,
    n_out: int = 1,
    hidden_layers: Sequence[int] = (32, 32),
    options: Optional[Dict[str, Any]] = None,
) -> nn.Module:
    """Build a task model on top of a reusable representation."""

    if not isinstance(
        representation,
        Representation,
    ):
        raise TypeError(
            "`representation` must derive from `Representation`."
        )

    if head is None:
        head = TaskHead(
            in_features=representation.out_features,
            n_out=n_out,
            hidden_layers=hidden_layers,
            options=options,
        )

    else:
        if (
            not hasattr(head, "in_features")
            or not hasattr(head, "out_features")
        ):
            raise TypeError(
                "`head` must expose `in_features` and `out_features`."
            )

        if (
            int(head.in_features)
            != representation.out_features
        ):
            raise ValueError(
                "`head.in_features` does not match "
                "the representation output: "
                f"expected {representation.out_features}, "
                f"found {head.in_features}."
            )

    if representation.input_kind == "tensor":
        return _TensorRepresentationModel(
            representation=representation,
            head=head,
        )

    if not isinstance(
        representation,
        GraphRepresentation,
    ):
        raise TypeError(
            "Graph representations must derive from "
            "`GraphRepresentation`."
        )

    return _GraphRepresentationModel(
        representation=representation,
        head=head,
    )