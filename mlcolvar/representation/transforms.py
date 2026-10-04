from collections.abc import Sequence

import torch

from ._utils import as_float
from .base import Representation

__all__ = ["SelectAtoms"]


class SelectAtoms(Representation):
    """Select and concatenate fixed atom-level representation features.

    The wrapped representation must return atom-level features. Features of the
    selected atoms are concatenated for each system, assuming consistent atom ordering.

    Parameters
    ----------
    representation : Representation
        Atom-level graph representation.
    atom_indices : Sequence[int]
        Non-negative, unique atom indices to select from each system.
    """

    __constants__ = ["max_atom_index"]

    def __init__(
        self,
        representation: Representation,
        atom_indices: Sequence[int],
    ) -> None:
        if representation.input_kind != "graph":
            raise ValueError("`representation` must use graph inputs.")
        if representation.output_kind != "atom":
            raise ValueError("`representation` must return atom-level features.")

        indices = [int(index) for index in atom_indices]
        if not indices or any(index < 0 for index in indices):
            raise ValueError("`atom_indices` must contain non-negative indices.")
        if len(indices) != len(set(indices)):
            raise ValueError("`atom_indices` must not contain duplicates.")

        super().__init__(
            out_features=representation.out_features * len(indices),
            input_kind="graph",
            output_kind="system",
            atomic_numbers=representation.atomic_numbers,
            cutoff=as_float(representation.cutoff, "representation.cutoff"),
            buffer=as_float(representation.buffer, "representation.buffer"),
            long_range_cutoff=as_float(
                representation.long_range_cutoff,
                "representation.long_range_cutoff",
            ),
            freeze=representation.freeze,
        )

        self.representation = representation
        self.max_atom_index = max(indices)
        self.register_buffer("atom_indices", torch.tensor(indices, dtype=torch.long))

    def forward(
        self,
        data: dict[str, torch.Tensor],
        cell: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Select atom features and concatenate them for each system."""
        features = self.representation(data, cell=cell)
        ptr = data["ptr"].to(features.device, torch.long)

        if features.ndim != 2:
            raise ValueError(
                "The wrapped representation must return atom-level features."
            )

        atoms_per_system = ptr[1:] - ptr[:-1]
        if torch.any(atoms_per_system <= self.max_atom_index):
            raise RuntimeError(
                "A selected atom index exceeds the number of atoms "
                "in at least one system."
            )

        indices = ptr[:-1].unsqueeze(1) + self.atom_indices.to(
            features.device
        ).unsqueeze(0)

        selected = features.index_select(0, indices.reshape(-1))
        return selected.reshape(ptr.numel() - 1, -1)