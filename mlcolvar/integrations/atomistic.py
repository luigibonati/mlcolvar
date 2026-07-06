"""Common interfaces for external atomistic-model integrations."""

from typing import Dict, List, Optional, Tuple

import torch
from torch import nn

from mlcolvar.core import BaseGNN, FeedForward


__all__ = [
    "BaseAtomisticFeaturizer",
    "AtomisticModel",
]


class BaseAtomisticFeaturizer(nn.Module):
    """Base class for external atomistic-model featurizers.

    External integrations such as MACE, NequIP, DeepMD, and metatomic
    should inherit from this class and implement :meth:`forward`.

    Implementations are expected to map an mlcolvar graph dictionary to
    graph-level features with shape ``[n_graphs, out_features]``.

    Parameters
    ----------
    out_features
        Number of graph-level features returned by the featurizer.
    atomic_numbers
        Atomic numbers supported by the external model. Their order must
        match the species encoding used to construct ``node_attrs``.
    cutoff
        Short-range cutoff required by the external model.
    pooling
        Operation used to convert node-level features into graph-level
        features. Available options are ``"mean"`` and ``"sum"``.
    buffer
        Additional environment buffer used when constructing graphs.
    long_range_cutoff
        Optional cutoff for long-range edges. A negative value disables
        long-range edges.
    """

    __constants__ = [
        "out_features",
        "pooling",
    ]

    def __init__(
        self,
        out_features: int,
        atomic_numbers: List[int],
        cutoff: float,
        pooling: str = "mean",
        buffer: float = 0.0,
        long_range_cutoff: float = -1.0,
    ) -> None:
        super().__init__()

        if out_features <= 0:
            raise ValueError(
                f"out_features must be positive, found {out_features}."
            )

        if len(atomic_numbers) == 0:
            raise ValueError(
                "atomic_numbers cannot be empty."
            )

        if len(set(atomic_numbers)) != len(atomic_numbers):
            raise ValueError(
                "atomic_numbers must not contain duplicates: "
                f"{atomic_numbers}."
            )

        if any(number <= 0 for number in atomic_numbers):
            raise ValueError(
                "atomic_numbers must contain positive integers: "
                f"{atomic_numbers}."
            )

        if cutoff <= 0:
            raise ValueError(
                f"cutoff must be positive, found {cutoff}."
            )

        if buffer < 0:
            raise ValueError(
                f"buffer must be non-negative, found {buffer}."
            )

        if (
            long_range_cutoff >= 0
            and long_range_cutoff <= cutoff
        ):
            raise ValueError(
                "long_range_cutoff must be negative or larger than "
                f"cutoff. Found cutoff={cutoff} and "
                f"long_range_cutoff={long_range_cutoff}."
            )

        if pooling not in ("mean", "sum"):
            raise ValueError(
                "pooling must be either 'mean' or 'sum', "
                f"found {pooling!r}."
            )

        self.out_features = int(out_features)
        self.pooling = pooling

        # Serialized metadata required by graph construction and export.
        self.register_buffer(
            "feature_dim",
            torch.tensor(
                out_features,
                dtype=torch.int64,
            ),
        )

        self.register_buffer(
            "atomic_numbers",
            torch.tensor(
                atomic_numbers,
                dtype=torch.int64,
            ),
        )

        self.register_buffer(
            "cutoff",
            torch.tensor(
                cutoff,
                dtype=torch.get_default_dtype(),
            ),
        )

        self.register_buffer(
            "buffer",
            torch.tensor(
                buffer,
                dtype=torch.get_default_dtype(),
            ),
        )

        self.register_buffer(
            "long_range_cutoff",
            torch.tensor(
                long_range_cutoff,
                dtype=torch.get_default_dtype(),
            ),
        )

    @property
    def in_features(self) -> Optional[int]:
        """Atomistic featurizers receive graphs rather than flat tensors."""
        return None

    def forward(
        self,
        data: Dict[str, torch.Tensor],
        cell: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute graph-level features.

        Subclasses must implement this method.

        Parameters
        ----------
        data
            mlcolvar graph dictionary.
        cell
            Optional simulation cell passed separately by the mlcolvar
            CV interface.

        Returns
        -------
        torch.Tensor
            Graph-level features with shape
            ``[n_graphs, out_features]``.
        """
        raise NotImplementedError

    def _pool_node_features(
        self,
        node_features: torch.Tensor,
        data: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Pool node-level features into graph-level features."""

        batch = data["batch"].to(
            device=node_features.device,
            dtype=torch.long,
        )

        n_graphs = self._get_num_graphs(
            data
        )

        if "system_masks" in data:
            mask = data["system_masks"].reshape(-1, 1).to(
                device=node_features.device,
                dtype=node_features.dtype,
            )
        else:
            mask = torch.ones(
                (node_features.shape[0], 1),
                dtype=node_features.dtype,
                device=node_features.device,
            )

        masked_features = node_features * mask

        output = node_features.new_zeros(
            (
                n_graphs,
                node_features.shape[-1],
            )
        )

        output.index_add_(
            0,
            batch,
            masked_features,
        )

        if self.pooling == "sum":
            return output

        counts = node_features.new_zeros(
            (n_graphs, 1)
        )

        counts.index_add_(
            0,
            batch,
            mask,
        )

        return output / counts.clamp_min(1.0)

    def _get_num_graphs(
        self,
        data: Dict[str, torch.Tensor],
    ) -> int:
        """Infer the number of graphs in a batch."""

        if "ptr" in data:
            return data["ptr"].size(0) - 1

        if "n_system" in data:
            return data["n_system"].size(0)

        raise RuntimeError(
            "Cannot infer the number of graphs. The graph data must "
            "contain `ptr` or `n_system`."
        )


class AtomisticModel(BaseGNN):
    """External atomistic featurizer followed by a trainable CV readout.

    This generic wrapper allows any ``BaseAtomisticFeaturizer`` to be used
    as a graph model by mlcolvar CV classes without requiring one model
    wrapper for each external backend.

    Parameters
    ----------
    featurizer
        External atomistic-model featurizer.
    n_out
        Number of output collective variables.
    hidden_layers
        Hidden dimensions of the trainable feed-forward readout.
    """

    def __init__(
        self,
        featurizer: BaseAtomisticFeaturizer,
        n_out: int = 1,
        hidden_layers: Tuple[int, ...] = (30, 30),
    ) -> None:
        if n_out <= 0:
            raise ValueError(
                f"n_out must be positive, found {n_out}."
            )

        atomic_numbers = (
            featurizer.atomic_numbers
            .detach()
            .cpu()
            .tolist()
        )

        cutoff = float(
            featurizer.cutoff
            .detach()
            .cpu()
            .item()
        )

        buffer = float(
            featurizer.buffer
            .detach()
            .cpu()
            .item()
        )

        long_range_cutoff = float(
            featurizer.long_range_cutoff
            .detach()
            .cpu()
            .item()
        )

        # The featurizer already performs node-to-graph pooling.
        super().__init__(
            n_out=n_out,
            dataset_for_initialization=None,
            pooling_operation=None,
            cutoff=cutoff,
            buffer=buffer,
            long_range_cutoff=long_range_cutoff,
            atomic_numbers=atomic_numbers,
        )

        self.featurizer = featurizer

        self.readout = FeedForward(
            layers=[
                featurizer.out_features,
                *hidden_layers,
                n_out,
            ],
        )

    def forward(
        self,
        data: Dict[str, torch.Tensor],
        cell: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute collective variables from an atomic graph."""

        features = self.featurizer(
            data,
            cell=cell,
        )

        return self.readout(
            features
        )