from typing import Any, Dict, Literal, Optional, Sequence

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
    """Shared representation → reducer → head pipeline."""

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

    @torch.jit.unused
    def cache(
        self,
        dataset,
        *,
        jacobian: bool = False,
        **kwargs,
    ):
        """Precompute frozen features, optionally with coordinate Jacobians."""
        from .cache import precompute_representation_cache

        return precompute_representation_cache(
            self.representation,
            dataset,
            reducer=self.pre_head,
            compute_jacobian=jacobian,
            **kwargs,
        )


class _TensorRepresentationModel(_RepresentationPipelineMixin, nn.Module):
    """Complete model for tensor-input representations."""

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

        return self.post_head(
            self.head(features),
            x,
        )


class _GraphRepresentationModel(_RepresentationPipelineMixin, BaseGNN):
    """Complete model for graph-input representations."""

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

        # Representation adapters construct their own neighborhood features.
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

        return self.post_head(
            self.head(features),
            data,
        )


def _make_reducers(
    representation: Representation,
    mode: Optional[Literal["direct", "pooled", "nodewise", "concat"]],
    selected_atom_indices: Optional[Sequence[int]],
    pooling: str,
    n_out: int,
) -> tuple[Reducer, Reducer]:
    mode = mode or (
        "pooled"
        if representation.output_kind == "atom"
        else "direct"
    )

    allowed = {"direct", "pooled", "nodewise", "concat"}
    if mode not in allowed:
        raise ValueError(
            f"`mode` must be one of {sorted(allowed)}. Found {mode!r}."
        )

    if mode == "direct":
        if representation.output_kind != "system":
            raise ValueError(
                "`mode='direct'` requires a system-level representation."
            )

        return (
            IdentityReducer(representation.out_features),
            IdentityReducer(n_out),
        )

    if representation.output_kind != "atom":
        raise ValueError(
            f"`mode={mode!r}` requires an atom-level representation."
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
    mode: Optional[
        Literal["direct", "pooled", "nodewise", "concat"]
    ] = None,
    pooling: Literal["mean", "sum"] = "mean",
    selected_atom_indices: Optional[Sequence[int]] = None,
) -> nn.Module:
    """Build a task model on top of a reusable representation.

    Modes
    -----
    direct
        System features → task head.
    pooled
        Atom features → pooling → task head.
    nodewise
        Atom features → task head → pooling.
    concat
        Selected atom features → concatenation → task head.

    If ``mode`` is omitted, system-level representations use ``direct`` and
    atom-level representations use ``pooled``.

    Parameters
    ----------
    representation
        Reusable tensor or graph representation.
    head
        Optional custom task head. By default, a :class:`TaskHead` is created.
    n_out
        Number of model outputs.
    hidden_layers
        Hidden layers of the default :class:`TaskHead`.
    mode
        How atom/system features are connected to the task head.
    pooling
        ``"mean"`` or ``"sum"`` pooling.
    selected_atom_indices
        Atom indices used by ``mode="concat"``.
    """
    if not isinstance(representation, Representation):
        raise TypeError(
            "`representation` must derive from `Representation`."
        )

    if head is not None:
        if not hasattr(head, "in_features") or not hasattr(head, "out_features"):
            raise TypeError(
                "`head` must expose `in_features` and `out_features`."
            )

        n_out = int(head.out_features)

    pre_head, post_head = _make_reducers(
        representation,
        mode,
        selected_atom_indices,
        pooling,
        n_out,
    )

    if head is None:
        head = TaskHead(
            in_features=pre_head.out_features,
            n_out=n_out,
            hidden_layers=hidden_layers,
            options=options,
        )

    elif int(head.in_features) != pre_head.out_features:
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
            "Graph representations must derive from `GraphRepresentation`."
        )

    return _GraphRepresentationModel(
        representation=representation,
        pre_head=pre_head,
        head=head,
        post_head=post_head,
    )