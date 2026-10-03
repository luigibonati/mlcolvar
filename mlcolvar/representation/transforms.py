from collections.abc import Sequence

import torch
from torch import nn

__all__ = ["SelectAtoms"]


class SelectAtoms(nn.Module):
    """Select and concatenate atom-level features for each system."""

    __constants__ = ["max_atom_index"]

    def __init__(
        self,
        atom_indices: Sequence[int],
    ) -> None:
        super().__init__()

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

        self.max_atom_index = max(indices)
        self.register_buffer(
            "atom_indices",
            torch.tensor(indices, dtype=torch.long),
        )

    def forward(
        self,
        features: torch.Tensor,
        ptr: torch.Tensor,
    ) -> torch.Tensor:
        """Select atom features and concatenate them per system."""
        if features.ndim != 2:
            raise ValueError(
                "`features` must have shape (n_atoms, n_features)."
            )
        if ptr.ndim != 1:
            raise ValueError("`ptr` must be one-dimensional.")
        if ptr.numel() < 2:
            raise ValueError("`ptr` must describe at least one system.")

        ptr = ptr.to(
            device=features.device,
            dtype=torch.long,
        )

        atoms_per_system = ptr[1:] - ptr[:-1]
        if torch.any(atoms_per_system <= self.max_atom_index):
            raise RuntimeError(
                "A selected atom index exceeds the number of atoms "
                "in at least one system."
            )

        indices = (
            ptr[:-1].unsqueeze(1)
            + self.atom_indices.to(features.device).unsqueeze(0)
        )
        selected = features.index_select(
            0,
            indices.reshape(-1),
        )

        return selected.reshape(
            ptr.numel() - 1,
            -1,
        )