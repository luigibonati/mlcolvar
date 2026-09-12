from typing import Any, Dict, Sequence

import torch
from torch import nn

from .base import infer_num_graphs


__all__ = [
    "Reducer",
    "IdentityReducer",
    "PoolReducer",
    "ConcatReducer",
]


class Reducer(nn.Module):
    """Base feature transformation used before or after a task head."""

    __constants__ = ["in_features", "out_features"]

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)


class IdentityReducer(Reducer):
    def __init__(self, in_features: int) -> None:
        super().__init__(in_features, in_features)

    def forward(
        self,
        features: torch.Tensor,
        data: Any = None,
    ) -> torch.Tensor:
        return features


class PoolReducer(Reducer):
    """Pool atom-level features to one vector per system."""

    __constants__ = ["pooling"]

    def __init__(self, in_features: int, pooling: str = "mean") -> None:
        if pooling not in {"mean", "sum"}:
            raise ValueError("`pooling` must be 'mean' or 'sum'.")

        super().__init__(in_features, in_features)
        self.pooling = pooling

    def forward(
        self,
        features: torch.Tensor,
        data: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        if "batch" not in data:
            raise KeyError("Graph data must contain `batch` for pooling.")

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
        output.index_add_(
            0,
            batch,
            features * mask,
        )

        if self.pooling == "sum":
            return output

        counts = features.new_zeros(
            n_systems,
            1,
        )
        counts.index_add_(0, batch, mask)

        return output / counts.clamp_min(1)


class ConcatReducer(Reducer):
    """Select the same local atom indices in every graph and concatenate."""

    __constants__ = [
        "n_selected_atoms",
        "max_selected_atom_index",
    ]

    def __init__(
        self,
        in_features: int,
        atom_indices: Sequence[int],
    ) -> None:
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
            in_features,
            in_features * self.n_selected_atoms,
        )

        self.register_buffer(
            "atom_indices",
            torch.tensor(indices, dtype=torch.long),
        )

    def forward(
        self,
        features: torch.Tensor,
        data: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
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