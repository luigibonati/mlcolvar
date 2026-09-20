from typing import Any, Dict, Optional, Sequence

import torch
from torch import nn

from mlcolvar.core import BaseGNN, FeedForward

from .base import (
    GraphRepresentation,
    Representation,
    VectorRepresentation,
    infer_num_graphs,
    module_reference_tensor,
)


__all__ = [
    "TaskHead",
    "RepresentationModel",
    "pool_representation",
    "concat_representation",
]


def _graph_metadata(
    representation: GraphRepresentation,
    out_features: int,
) -> Dict[str, Any]:
    """Reuse graph metadata for a transformed representation."""

    return {
        "out_features": int(out_features),
        "atomic_numbers": representation.atomic_numbers.detach().cpu().tolist(),
        "cutoff": float(representation.cutoff.detach().cpu().item()),
        "output_kind": "system",
        "buffer": float(representation.buffer.detach().cpu().item()),
        "long_range_cutoff": float(
            representation.long_range_cutoff.detach().cpu().item()
        ),
        "full_neighbor_list": representation.full_neighbor_list,
        "freeze": representation.freeze,
    }


class _PooledGraphRepresentation(GraphRepresentation):
    """Pool atom-level features to one vector per system."""

    __constants__ = ["pooling"]

    def __init__(
        self,
        representation: GraphRepresentation,
        pooling: str = "mean",
    ) -> None:
        if representation.output_kind != "atom":
            raise ValueError(
                "`representation` must produce atom-level features."
            )

        if pooling not in {"mean", "sum"}:
            raise ValueError(
                "`pooling` must be 'mean' or 'sum'."
            )

        super().__init__(
            **_graph_metadata(
                representation,
                representation.out_features,
            )
        )

        self.representation = representation
        self.pooling = pooling

    def forward(
        self,
        data: Dict[str, torch.Tensor],
        cell: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        features = self.representation(data, cell=cell)

        if "batch" not in data:
            raise KeyError(
                "Graph data must contain `batch` for pooling."
            )

        batch = data["batch"].to(
            device=features.device,
            dtype=torch.long,
        )
        n_systems = infer_num_graphs(data)

        mask = (
            data["system_masks"].reshape(-1, 1).to(features)
            if "system_masks" in data
            else features.new_ones((features.size(0), 1))
        )

        output = features.new_zeros(
            n_systems,
            features.size(-1),
        )
        output.index_add_(0, batch, features * mask)

        if self.pooling == "sum":
            return output

        counts = features.new_zeros(n_systems, 1)
        counts.index_add_(0, batch, mask)

        return output / counts.clamp_min(1)


class _ConcatGraphRepresentation(GraphRepresentation):
    """Concatenate selected atom features for each system."""

    __constants__ = [
        "n_selected_atoms",
        "max_selected_atom_index",
    ]

    def __init__(
        self,
        representation: GraphRepresentation,
        atom_indices: Sequence[int],
    ) -> None:
        if representation.output_kind != "atom":
            raise ValueError(
                "`representation` must produce atom-level features."
            )

        indices = [int(index) for index in atom_indices]

        if not indices:
            raise ValueError(
                "`atom_indices` cannot be empty."
            )

        if any(index < 0 for index in indices):
            raise ValueError(
                "`atom_indices` must contain non-negative indices."
            )

        if len(indices) != len(set(indices)):
            raise ValueError(
                "`atom_indices` must not contain duplicates."
            )

        self.n_selected_atoms = len(indices)
        self.max_selected_atom_index = max(indices)

        super().__init__(
            **_graph_metadata(
                representation,
                representation.out_features * self.n_selected_atoms,
            )
        )

        self.representation = representation

        self.register_buffer(
            "atom_indices",
            torch.tensor(indices, dtype=torch.long),
        )

    def forward(
        self,
        data: Dict[str, torch.Tensor],
        cell: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        features = self.representation(data, cell=cell)

        if "ptr" not in data:
            raise KeyError(
                "Graph data must contain `ptr` for "
                "selected-atom concatenation."
            )

        ptr = data["ptr"].to(
            device=features.device,
            dtype=torch.long,
        )

        atoms_per_graph = ptr[1:] - ptr[:-1]

        if torch.any(
            atoms_per_graph <= self.max_selected_atom_index
        ):
            raise RuntimeError(
                "A selected atom index exceeds the number "
                "of atoms in at least one system."
            )

        atom_indices = self.atom_indices.to(
            device=features.device
        )

        global_indices = (
            ptr[:-1].unsqueeze(1)
            + atom_indices.unsqueeze(0)
        )

        selected = features.index_select(
            0,
            global_indices.reshape(-1),
        )

        return selected.reshape(
            ptr.numel() - 1,
            self.out_features,
        )


def pool_representation(
    representation: GraphRepresentation,
    pooling: str = "mean",
) -> GraphRepresentation:
    """Pool atom-level features into a system-level representation."""

    return _PooledGraphRepresentation(
        representation,
        pooling=pooling,
    )


def concat_representation(
    representation: GraphRepresentation,
    atom_indices: Sequence[int],
) -> GraphRepresentation:
    """Concatenate selected atoms into a system-level representation."""

    return _ConcatGraphRepresentation(
        representation,
        atom_indices=atom_indices,
    )


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
            **({} if options is None else dict(options)),
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
        return self.representation.cache(
            dataset,
            jacobian=jacobian,
            **kwargs,
        )


class _VectorRepresentationModel(
    _RepresentationPipelineMixin,
    nn.Module,
):
    """Task model for vector-input representations."""

    def __init__(
        self,
        *,
        representation: VectorRepresentation,
        head: nn.Module,
    ) -> None:
        nn.Module.__init__(self)

        self._init_pipeline(
            representation=representation,
            head=head,
        )

        self.in_features = int(representation.in_features)
        self.out_features = int(head.out_features)

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
    """Task model for graph-input representations."""

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
                representation.cutoff.detach().cpu().item()
            ),
            buffer=float(
                representation.buffer.detach().cpu().item()
            ),
            long_range_cutoff=float(
                representation.long_range_cutoff.detach().cpu().item()
            ),
            atomic_numbers=(
                representation.atomic_numbers.detach().cpu().tolist()
            ),
        )

        # Representation adapters construct their own neighborhood features.
        self._modules.pop("_radial_embedding", None)

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

    if not isinstance(representation, Representation):
        raise TypeError(
            "`representation` must derive from `Representation`."
        )

    if (
        isinstance(representation, GraphRepresentation)
        and representation.output_kind != "system"
    ):
        raise ValueError(
            "Graph representations passed to `RepresentationModel` "
            "must produce system-level features. Use "
            "`pool_representation` or `concat_representation` first."
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

        if int(head.in_features) != representation.out_features:
            raise ValueError(
                "`head.in_features` does not match "
                "the representation output: "
                f"expected {representation.out_features}, "
                f"found {head.in_features}."
            )

    if isinstance(representation, VectorRepresentation):
        return _VectorRepresentationModel(
            representation=representation,
            head=head,
        )

    if isinstance(representation, GraphRepresentation):
        return _GraphRepresentationModel(
            representation=representation,
            head=head,
        )

    raise TypeError(
        "Unsupported representation type."
    )