from typing import Sequence

import torch
from torch import nn

from ._utils import (
    _as_atomic_number_list,
    align_node_attrs,
    as_positive_int,
    infer_num_graphs,
    module_reference_tensor,
)


__all__ = [
    "Representation",
    "VectorRepresentation",
    "GraphRepresentation",
    "as_positive_int",
    "module_reference_tensor",
    "infer_num_graphs",
    "align_node_attrs",
]


class Representation(nn.Module):
    """Base class for reusable frozen or trainable representations.

    A representation maps raw model input to reusable latent features.
    It deliberately contains no task-specific readout.
    """

    __constants__ = [
        "input_kind",
        "output_kind",
        "out_features",
        "freeze",
    ]

    def __init__(
        self,
        *,
        out_features: int,
        input_kind: str,
        output_kind: str,
        freeze: bool = True,
    ) -> None:
        super().__init__()

        if input_kind not in {"vector", "graph"}:
            raise ValueError(
                "`input_kind` must be 'vector' or 'graph'."
            )

        if output_kind not in {"atom", "system"}:
            raise ValueError(
                "`output_kind` must be 'atom' or 'system'."
            )

        self.out_features = as_positive_int(
            out_features,
            "out_features",
        )
        self.input_kind = input_kind
        self.output_kind = output_kind
        self.freeze = bool(freeze)

    def _freeze_module(
        self,
        module: nn.Module,
    ) -> None:
        if not self.freeze:
            return

        if not isinstance(
            module,
            torch.jit.ScriptModule,
        ):
            module.requires_grad_(False)

        module.eval()

    def train(
        self,
        mode: bool = True,
    ):
        """Set training mode while keeping frozen representations in eval mode."""
        return super().train(
            False if self.freeze else mode
        )

    @torch.jit.unused
    def cache(
        self,
        dataset,
        *,
        jacobian: bool = False,
        **kwargs,
    ):
        """Precompute and cache representation features.

        Parameters
        ----------
        dataset
            Dataset used to evaluate the representation.
        jacobian
            If True, also cache derivatives of the representation.
        **kwargs
            Additional arguments forwarded to the cache implementation.

        Returns
        -------
        RepresentationCache
            Cached representation features and, optionally, Jacobians.

        Notes
        -----
        Feature-only caching supports variable-size graph systems.

        Graph caching expects one system-level feature vector per graph.
        Atom-level graph representations should first be transformed with
        :meth:`GraphRepresentation.pool` or
        :meth:`GraphRepresentation.concat_atoms`.

        When ``jacobian=True`` for a graph representation, all selected
        graphs must contain the same number of atoms because coordinate
        Jacobians are stored in a dense tensor.
        """
        if not self.freeze:
            raise RuntimeError(
                "Caching requires a frozen representation."
            )

        # Local import avoids a circular dependency.
        from .cache import precompute_representation_cache

        return precompute_representation_cache(
            self,
            dataset,
            compute_jacobian=jacobian,
            **kwargs,
        )


class VectorRepresentation(Representation):
    """Base representation accepting dense feature vectors."""

    __constants__ = ["in_features"]

    def __init__(
        self,
        *,
        in_features: int,
        out_features: int,
        output_kind: str = "system",
        freeze: bool = True,
    ) -> None:
        super().__init__(
            out_features=out_features,
            input_kind="vector",
            output_kind=output_kind,
            freeze=freeze,
        )

        self.in_features = as_positive_int(
            in_features,
            "in_features",
        )


class GraphRepresentation(Representation):
    """Base representation accepting mlcolvar graph dictionaries."""

    __constants__ = [
        "full_neighbor_list",
    ]

    def __init__(
        self,
        *,
        out_features: int,
        atomic_numbers: Sequence[int] | torch.Tensor,
        cutoff: float,
        output_kind: str = "atom",
        buffer: float = 0.0,
        long_range_cutoff: float = -1.0,
        full_neighbor_list: bool = True,
        freeze: bool = True,
    ) -> None:
        atomic_numbers = _as_atomic_number_list(
            atomic_numbers,
            "representation",
        )

        if cutoff <= 0.0:
            raise ValueError(
                "`cutoff` must be positive."
            )

        if buffer < 0.0:
            raise ValueError(
                "`buffer` must be non-negative."
            )

        if 0.0 <= long_range_cutoff <= cutoff:
            raise ValueError(
                "`long_range_cutoff` must be negative "
                "or larger than `cutoff`."
            )

        super().__init__(
            out_features=out_features,
            input_kind="graph",
            output_kind=output_kind,
            freeze=freeze,
        )

        # BaseCV uses in_features=None to identify graph models.
        self.in_features = None
        self.full_neighbor_list = bool(
            full_neighbor_list
        )

        self.register_buffer(
            "feature_dim",
            torch.tensor(
                self.out_features,
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

    @torch.jit.unused
    def align_dataset(
        self,
        dataset,
    ):
        """Align dataset atomic species with the representation.

        The graph ``node_attrs`` and ``atomic_numbers`` metadata are updated
        in place to match the atomic-number ordering expected by the
        representation. The input dataset is returned for convenience.
        """
        return align_node_attrs(
            dataset,
            self.atomic_numbers,
        )

    @torch.jit.unused
    def pool(
        self,
        pooling: str = "mean",
    ) -> "GraphRepresentation":
        """Pool atom-level features into system-level features."""
        from .model import pool_representation

        return pool_representation(
            self,
            pooling=pooling,
        )

    @torch.jit.unused
    def concat_atoms(
        self,
        atom_indices: Sequence[int],
    ) -> "GraphRepresentation":
        """Concatenate features from selected atoms.

        Parameters
        ----------
        atom_indices
            Zero-based atom indices local to each graph. The same atom
            indices are selected independently from every graph in the batch.

        Returns
        -------
        GraphRepresentation
            System-level representation obtained by concatenating the
            selected atom features.
        """
        from .model import concat_representation

        return concat_representation(
            self,
            atom_indices=atom_indices,
        )