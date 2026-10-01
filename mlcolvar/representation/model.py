from typing import Any, Dict, Optional, Sequence

import torch
from torch import nn

from mlcolvar.core import BaseGNN, FeedForward

from .base import GraphRepresentation, Representation, VectorRepresentation
from ._utils import module_reference_tensor

__all__ = ["RepresentationModel"]


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
            raise ValueError("`atom_indices` cannot be empty.")
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
            out_features=representation.out_features * self.n_selected_atoms,
            atomic_numbers=representation.atomic_numbers.detach().cpu().tolist(),
            cutoff=float(representation.cutoff.detach().cpu().item()),
            pooling_operation=None,
            output_kind="system",
            buffer=float(representation.buffer.detach().cpu().item()),
            long_range_cutoff=float(
                representation.long_range_cutoff.detach().cpu().item()
            ),
            full_neighbor_list=representation.full_neighbor_list,
            freeze=representation.freeze,
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

        if torch.any(atoms_per_graph <= self.max_selected_atom_index):
            raise RuntimeError(
                "A selected atom index exceeds the number "
                "of atoms in at least one system."
            )

        global_indices = (
            ptr[:-1].unsqueeze(1)
            + self.atom_indices.to(features.device).unsqueeze(0)
        )
        selected = features.index_select(
            0,
            global_indices.reshape(-1),
        )

        return selected.reshape(
            ptr.numel() - 1,
            self.out_features,
        )


def concat_representation(
    representation: GraphRepresentation,
    atom_indices: Sequence[int],
) -> GraphRepresentation:
    """Concatenate selected atom features into a system-level representation."""
    return _ConcatGraphRepresentation(
        representation,
        atom_indices,
    )


class _RepresentationPipelineMixin:
    """Shared representation → head pipeline."""

    def _init_pipeline(
        self,
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

    def _apply_head(self, features):
        features = features.to(
            device=self._head_reference.device,
            dtype=self._head_reference.dtype,
        )
        return self.head(features)

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
        representation: VectorRepresentation,
        head: nn.Module,
    ) -> None:
        nn.Module.__init__(self)
        self._init_pipeline(representation, head)
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

        return self._apply_head(
            self.representation(x, cell=cell)
        )


class _GraphRepresentationModel(
    _RepresentationPipelineMixin,
    BaseGNN,
):
    """Task model for graph-input representations."""

    def __init__(
        self,
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

        self._modules.pop("_radial_embedding", None)
        self._init_pipeline(representation, head)

    def forward(
        self,
        data: Dict[str, torch.Tensor],
        cell: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self._apply_head(
            self.representation(data, cell=cell)
        )


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
            "must produce system-level features. "
            "Set `pooling_operation='mean'` or `'sum'`, "
            "or use `.concat_atoms()` first."
        )

    if head is None:
        head = FeedForward(
            layers=[
                representation.out_features,
                *hidden_layers,
                n_out,
            ],
            **({} if options is None else dict(options)),
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
            representation,
            head,
        )

    if isinstance(representation, GraphRepresentation):
        return _GraphRepresentationModel(
            representation,
            head,
        )

    raise TypeError("Unsupported representation type.")