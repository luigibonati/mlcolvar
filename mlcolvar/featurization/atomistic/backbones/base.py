from typing import Dict, Optional, Sequence

import torch
from torch import nn


__all__ = [
    "BaseAtomisticBackbone",
]


class BaseAtomisticBackbone(nn.Module):
    """Base class for pretrained atomistic-model backbones.

    Parameters
    ----------
    out_features
        Number of features returned for each atom or system.
    atomic_numbers
        Atomic numbers supported by the pretrained model.
    cutoff
        Short-range interaction cutoff.
    sample_kind
        Whether the backbone returns one feature vector per ``"atom"``
        or per ``"system"``.
    buffer
        Additional neighbor-list buffer.
    long_range_cutoff
        Optional long-range cutoff. A negative value disables it.
    full_neighbor_list
        Whether the backbone requires a full neighbor list.
    """

    __constants__ = [
        "out_features",
        "sample_kind",
        "full_neighbor_list",
    ]

    def __init__(
        self,
        out_features: int,
        atomic_numbers: Sequence[int],
        cutoff: float,
        sample_kind: str = "atom",
        buffer: float = 0.0,
        long_range_cutoff: float = -1.0,
        full_neighbor_list: bool = True,
    ) -> None:
        super().__init__()

        atomic_numbers = [int(number) for number in atomic_numbers]

        if out_features <= 0:
            raise ValueError("`out_features` must be positive.")

        if len(atomic_numbers) == 0:
            raise ValueError("`atomic_numbers` cannot be empty.")

        if len(set(atomic_numbers)) != len(atomic_numbers):
            raise ValueError("`atomic_numbers` must not contain duplicates.")

        if any(number <= 0 for number in atomic_numbers):
            raise ValueError("`atomic_numbers` must contain positive integers.")

        if cutoff <= 0.0:
            raise ValueError("`cutoff` must be positive.")

        if buffer < 0.0:
            raise ValueError("`buffer` must be non-negative.")

        if long_range_cutoff >= 0.0 and long_range_cutoff <= cutoff:
            raise ValueError(
                "`long_range_cutoff` must be negative or larger than `cutoff`."
            )

        if sample_kind not in ("atom", "system"):
            raise ValueError("`sample_kind` must be either 'atom' or 'system'.")

        self.out_features = int(out_features)
        self.sample_kind = sample_kind
        self.full_neighbor_list = bool(full_neighbor_list)

        self.register_buffer(
            "feature_dim",
            torch.tensor(out_features, dtype=torch.int64),
        )
        self.register_buffer(
            "atomic_numbers",
            torch.tensor(atomic_numbers, dtype=torch.int64),
        )
        self.register_buffer(
            "cutoff",
            torch.tensor(cutoff, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "buffer",
            torch.tensor(buffer, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "long_range_cutoff",
            torch.tensor(long_range_cutoff, dtype=torch.get_default_dtype()),
        )

    def forward(
        self,
        data: Dict[str, torch.Tensor],
        cell: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return atom- or system-level features."""

        raise NotImplementedError
