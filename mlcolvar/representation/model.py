from typing import Any, Dict, Optional, Sequence

import torch
from torch import nn

from mlcolvar.core import BaseGNN, FeedForward

from .base import GraphRepresentation, Representation, module_reference_tensor
from .reducers import ConcatReducer, IdentityReducer, PoolReducer, Reducer


__all__ = [
    "TaskHead",
    "RepresentationModel",
]


class TaskHead(FeedForward):
    """Small task-specific MLP acting on a reusable representation."""

    def __init__(
        self,
        in_features: int,
        n_out: int = 1,
        hidden_layers: Sequence[int] = (32, 32),
        options: Optional[Dict[str, Any]] = None,
    ) -> None:
        in_features = int(in_features)
        n_out = int(n_out)
        hidden_layers = tuple(int(size) for size in hidden_layers)

        if in_features <= 0:
            raise ValueError("`in_features` must be positive.")
        if n_out <= 0:
            raise ValueError("`n_out` must be positive.")
        if any(size <= 0 for size in hidden_layers):
            raise ValueError("`hidden_layers` must contain positive integers.")

        super().__init__(
            layers=[in_features, *hidden_layers, n_out],
            **({} if options is None else dict(options)),
        )


class _RepresentationPipelineMixin:
    def _init_pipeline(
        self,
        *,
        representation: Representation,
        pre_head: Reducer,
        head: nn.Module,
        post_head: Reducer,
    ) -> None:
        self.representation = representation
        self.pre_head = pre_head
        self.head = head
        self.post_head = post_head
        self.register_buffer(
            "_head_reference",
            module_reference_tensor(head),
            persistent=False,
        )


class _TensorRepresentationModel(_RepresentationPipelineMixin, nn.Module):
    """Complete tensor-input representation + task-head model."""

    def __init__(
        self,
        *,
        representation: Representation,
        pre_head: Reducer,
        head: nn.Module,
        post_head: Reducer,
    ) -> None:
        nn.Module.__init__(self)
        self._init_pipeline(
            representation=representation,
            pre_head=pre_head,
            head=head,
            post_head=post_head,
        )
        self.in_features = int(representation.in_features)
        self.out_features = int(head.out_features)

    def forward(
        self,
        x: torch.Tensor,
        cell: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if x.ndim < 2:
            raise ValueError("`x` must include a batch dimension.")
        if x.shape[-1] != self.in_features:
            raise ValueError(
                f"Expected {self.in_features} input features, found {x.shape[-1]}."
            )

        features = self.representation(x, cell=cell)
        features = self.pre_head(features, x)
        features = features.to(
            device=self._head_reference.device,
            dtype=self._head_reference.dtype,
        )
        output = self.head(features)
        return self.post_head(output, x)


class _GraphRepresentationModel(_RepresentationPipelineMixin, BaseGNN):
    """Graph-input model with BaseGNN compatibility isolated in one class."""

    def __init__(
        self,
        *,
        representation: GraphRepresentation,
        pre_head: Reducer,
        head: nn.Module,
        post_head: Reducer,
    ) -> None:
        BaseGNN.__init__(
            self,
            n_out=int(head.out_features),
            dataset_for_initialization=None,
            pooling_operation=None,
            cutoff=float(representation.cutoff.detach().cpu().item()),
            buffer=float(representation.buffer.detach().cpu().item()),
            long_range_cutoff=float(
                representation.long_range_cutoff.detach().cpu().item()
            ),
            atomic_numbers=representation.atomic_numbers.detach().cpu().tolist(),
        )

        # Compatibility only: representation adapters construct their own
        # neighborhood features, therefore BaseGNN's radial embedding is unused.
        self._modules.pop("_radial_embedding", None)

        self._init_pipeline(
            representation=representation,
            pre_head=pre_head,
            head=head,
            post_head=post_head,
        )

    def forward(
        self,
        data: Dict[str, torch.Tensor],
        cell: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        features = self.representation(data, cell=cell)
        features = self.pre_head(features, data)
        features = features.to(
            device=self._head_reference.device,
            dtype=self._head_reference.dtype,
        )
        output = self.head(features)
        return self.post_head(output, data)


def _make_reducers(
    representation: Representation,
    mode: Optional[str],
    selected_atom_indices: Optional[Sequence[int]],
    pooling: str,
    n_out: int,
) -> tuple[Reducer, Reducer]:
    if mode is None:
        mode = "pooled" if representation.output_kind == "atom" else "direct"

    mode = mode.lower()
    allowed = {"direct", "pooled", "nodewise", "concat"}

    if mode not in allowed:
        raise ValueError(
            f"`mode` must be one of {sorted(allowed)}. Found {mode!r}."
        )

    if representation.output_kind == "system" and mode != "direct":
        raise ValueError(
            f"`mode={mode!r}` requires an atom-level representation."
        )

    if mode == "direct":
        return (
            IdentityReducer(representation.out_features),
            IdentityReducer(n_out),
        )

    if mode == "pooled":
        return (
            PoolReducer(
                representation.out_features,
                pooling=pooling,
            ),
            IdentityReducer(n_out),
        )

    if mode == "nodewise":
        return (
            IdentityReducer(representation.out_features),
            PoolReducer(
                n_out,
                pooling=pooling,
            ),
        )

    if selected_atom_indices is None:
        raise ValueError(
            "`selected_atom_indices` is required for mode='concat'."
        )

    return (
        ConcatReducer(
            representation.out_features,
            selected_atom_indices,
        ),
        IdentityReducer(n_out),
    )


def RepresentationModel(
    representation: Representation,
    *,
    head: Optional[nn.Module] = None,
    n_out: int = 1,
    hidden_layers: Sequence[int] = (32, 32),
    options: Optional[Dict[str, Any]] = None,
    mode: Optional[str] = None,
    pooling: str = "mean",
    selected_atom_indices: Optional[Sequence[int]] = None,
) -> nn.Module:
    """Build a complete model from any reusable representation.

    Parameters
    ----------
    representation
        Tensor or graph representation adapter.
    head
        Optional pre-built task head. If omitted, a :class:`TaskHead` is built.
    mode
        ``direct`` for system-level representations; ``pooled``, ``nodewise``
        or ``concat`` for atom-level representations. If omitted, system-level
        outputs use ``direct`` and atom-level outputs use ``pooled``.
    """
    if not isinstance(representation, Representation):
        raise TypeError(
            "`representation` must derive from `Representation`."
        )

    if head is None:
        temporary_pre, _ = _make_reducers(
            representation=representation,
            mode=mode,
            selected_atom_indices=selected_atom_indices,
            pooling=pooling,
            n_out=n_out,
        )

        head = TaskHead(
            in_features=temporary_pre.out_features,
            n_out=n_out,
            hidden_layers=hidden_layers,
            options=options,
        )

    else:
        if not hasattr(head, "in_features") or not hasattr(
            head,
            "out_features",
        ):
            raise TypeError(
                "`head` must expose `in_features` and `out_features`."
            )

        n_out = int(head.out_features)

    pre_head, post_head = _make_reducers(
        representation=representation,
        mode=mode,
        selected_atom_indices=selected_atom_indices,
        pooling=pooling,
        n_out=n_out,
    )

    if int(head.in_features) != pre_head.out_features:
        raise ValueError(
            "`head.in_features` does not match the representation/reducer output: "
            f"expected {pre_head.out_features}, found {head.in_features}."
        )

    if representation.input_kind == "tensor":
        return _TensorRepresentationModel(
            representation=representation,
            pre_head=pre_head,
            head=head,
            post_head=post_head,
        )

    if not isinstance(representation, GraphRepresentation):
        raise TypeError(
            "Graph representations must derive from `GraphRepresentation` so "
            "deployment metadata are available."
        )

    return _GraphRepresentationModel(
        representation=representation,
        pre_head=pre_head,
        head=head,
        post_head=post_head,
    )