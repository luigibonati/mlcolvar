
from collections.abc import Sequence

import torch

from ._utils import as_float
from .base import Representation

__all__ = ["SelectAtoms"]


class SelectAtoms(Representation):
    """Concatenate features of selected atoms from each graph.

    Parameters
    ----------
    representation : Representation
        Atom-level graph representation.
    atom_indices : Sequence[int]
        Unique, non-negative atom indices.
    """

    __constants__ = ["max_atom_index"]

    def __init__(
        self,
        representation: Representation,
        atom_indices: Sequence[int],
    ) -> None:
        if representation.input_kind != "graph" or representation.output_kind != "atom":
            raise ValueError("Expected an atom-level graph representation.")

        indices = [int(i) for i in atom_indices]
        if not indices or min(indices) < 0 or len(indices) != len(set(indices)):
            raise ValueError("`atom_indices` must be non-negative and unique.")

        super().__init__(
            out_features=representation.out_features * len(indices),
            input_kind="graph",
            output_kind="system",
            atomic_numbers=representation.atomic_numbers,
            cutoff=as_float(representation.cutoff, "cutoff"),
            buffer=as_float(representation.buffer, "buffer"),
            long_range_cutoff=as_float(
                representation.long_range_cutoff, "long_range_cutoff"
            ),
            freeze=representation.freeze,
        )

        self.representation = representation
        self.max_atom_index = max(indices)
        self.register_buffer(
            "atom_indices", torch.tensor(indices, dtype=torch.long)
        )

    def forward(
        self,
        data: dict[str, torch.Tensor],
        cell: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return selected atom features concatenated per graph."""
        features = self.representation(data, cell=cell)
        if features.ndim != 2:
            raise ValueError("Expected atom-level features with shape (N, F).")

        ptr = data["ptr"].to(features.device, dtype=torch.long)
        if torch.any(ptr[1:] - ptr[:-1] <= self.max_atom_index):
            raise RuntimeError("Selected atom index exceeds graph size.")

        indices = ptr[:-1, None] + self.atom_indices.to(features.device)
        return features.index_select(0, indices.flatten()).reshape(
            -1, self.out_features
        )
