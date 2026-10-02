from typing import Dict, Optional, Sequence

import torch
from torch import nn

from mlcolvar.utils import _code

from ._utils import (
    _as_atomic_number_list,
    align_node_attrs,
    as_positive_int,
)

__all__ = [
    "Representation",
    "VectorRepresentation",
    "GraphRepresentation",
]


class Representation(nn.Module):
    """Base class for reusable preprocessing representations.

    A representation maps raw model inputs to latent features and can be used
    directly as a ``BaseCV.preprocessing`` module. Frozen representations can
    also be materialized offline with :meth:`cache`.

    Parameters
    ----------
    out_features : int
        Number of output features.
    input_kind : {"vector", "graph"}
        Type of input expected by the representation.
    output_kind : {"atom", "system"}
        Whether the representation produces atom-level or system-level features.
    freeze : bool, default=True
        If True, keep the representation parameters frozen and in evaluation mode.
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
            raise ValueError("`input_kind` must be 'vector' or 'graph'.")
        if output_kind not in {"atom", "system"}:
            raise ValueError("`output_kind` must be 'atom' or 'system'.")
        self.out_features = as_positive_int(out_features, "out_features")
        self.input_kind = input_kind
        self.output_kind = output_kind
        self.freeze = bool(freeze)

    def _freeze_module(
        self,
        module: nn.Module,
    ) -> None:
        if not self.freeze:
            return
        if not isinstance(module, torch.jit.ScriptModule):
            module.requires_grad_(False)
        module.eval()

    def train(
        self,
        mode: bool = True,
    ):
        """Keep frozen representations in evaluation mode."""
        return super().train(False if self.freeze else mode)

    @torch.jit.unused
    def cache(
        self,
        dataset,
        *,
        jacobian: bool = False,
        **kwargs,
    ):
        """Materialize this frozen preprocessing on a dataset.

        Parameters
        ----------
        dataset
            Dataset containing the raw inputs.
        jacobian : bool, default=False
            If True, also materialize derivatives of the latent features.
        **kwargs
            Additional arguments forwarded to the cache implementation.

        Returns
        -------
        RepresentationCache
            Materialized latent features and, optionally, their Jacobians.

        Notes
        -----
        Graph caching requires system-level outputs. Use
        ``pooling_operation="mean"`` or ``"sum"``, or
        :meth:`GraphRepresentation.concat_atoms`.

        Graph Jacobians require selected systems to contain the same number of
        atoms because the derivatives are stored as a dense tensor.
        """
        if not self.freeze:
            raise RuntimeError("Caching requires a frozen representation.")

        from .cache import precompute_representation_cache

        return precompute_representation_cache(
            self,
            dataset,
            compute_jacobian=jacobian,
            **kwargs,
        )


class VectorRepresentation(Representation):
    """Base representation for dense vector inputs.

    Parameters
    ----------
    in_features : int
        Number of input features.
    out_features : int
        Number of output features.
    output_kind : {"atom", "system"}, default="system"
        Level of the representation output.
    freeze : bool, default=True
        If True, keep the representation parameters frozen and in evaluation mode.
    """

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
        self.in_features = as_positive_int(in_features, "in_features")


class GraphRepresentation(Representation):
    """Base representation for atomistic graph inputs.

    Parameters
    ----------
    out_features : int
        Number of output features per atom or system.
    atomic_numbers : sequence of int or torch.Tensor
        Atomic numbers supported by the representation.
    cutoff : float
        Cutoff used to construct the local neighbor list.
    pooling_operation : {"mean", "sum"}, optional
        Operation used to reduce atom-level features to system-level features.
        If None, atom-level features are returned.
    output_kind : {"atom", "system"}, optional
        Level of the output. If not provided, it is inferred from
        ``pooling_operation``.
    buffer : float, default=0.0
        Buffer added to the neighbor-list cutoff.
    long_range_cutoff : float, default=-1.0
        Optional cutoff for long-range interactions. A negative value disables it.
    full_neighbor_list : bool, default=True
        Whether the representation requires a full neighbor list.
    freeze : bool, default=True
        If True, keep the representation parameters frozen and in evaluation mode.
    """

    __constants__ = [
        "full_neighbor_list",
        "pooling_operation",
    ]

    def __init__(
        self,
        *,
        out_features: int,
        atomic_numbers: Sequence[int] | torch.Tensor,
        cutoff: float,
        pooling_operation: Optional[str] = None,
        output_kind: Optional[str] = None,
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
            raise ValueError("`cutoff` must be positive.")
        if buffer < 0.0:
            raise ValueError("`buffer` must be non-negative.")
        if 0.0 <= long_range_cutoff <= cutoff:
            raise ValueError(
                "`long_range_cutoff` must be negative or larger than `cutoff`."
            )
        if pooling_operation not in {None, "mean", "sum"}:
            raise ValueError(
                "`pooling_operation` must be 'mean', 'sum', or None."
            )
        if output_kind is None:
            output_kind = "atom" if pooling_operation is None else "system"

        super().__init__(
            out_features=out_features,
            input_kind="graph",
            output_kind=output_kind,
            freeze=freeze,
        )
        self.in_features = None
        self.pooling_operation = pooling_operation
        self.full_neighbor_list = bool(full_neighbor_list)
        self.register_buffer(
            "feature_dim",
            torch.tensor(self.out_features, dtype=torch.int64),
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
            torch.tensor(
                long_range_cutoff,
                dtype=torch.get_default_dtype(),
            ),
        )

    def pooling(
        self,
        input: torch.Tensor,
        data: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Pool atom-level features into system-level features.

        Parameters
        ----------
        input : torch.Tensor
            Atom-level features.
        data : dict
            Graph data containing batch indices and optional system masks.

        Returns
        -------
        torch.Tensor
            Atom-level features if no pooling is configured, otherwise pooled
            system-level features.
        """
        if self.pooling_operation is None:
            return input
        if self.pooling_operation == "mean":
            if "system_masks" not in data:
                return _code.scatter_mean(
                    input,
                    data["batch"],
                    dim=0,
                )
            output = input * data["system_masks"]
            output = _code.scatter_sum(
                output,
                data["batch"],
                dim=0,
            )
            return output / data["n_system"]
        if "system_masks" in data:
            input = input * data["system_masks"]
        return _code.scatter_sum(
            input,
            data["batch"],
            dim=0,
        )

    @torch.jit.unused
    def align_dataset(
        self,
        dataset,
    ):
        """Align graph node attributes with the representation atomic species.

        Parameters
        ----------
        dataset
            Graph dataset to align.

        Returns
        -------
        dataset
            Dataset with node attributes aligned with ``atomic_numbers``.
        """
        return align_node_attrs(
            dataset,
            self.atomic_numbers,
        )

    @torch.jit.unused
    def concat_atoms(
        self,
        atom_indices: Sequence[int],
    ) -> "GraphRepresentation":
        """Build a system-level representation from selected atom features.

        Parameters
        ----------
        atom_indices : sequence of int
            Atom indices to concatenate for each system.

        Returns
        -------
        GraphRepresentation
            System-level representation with
            ``out_features * len(atom_indices)`` output features.
        """
        return _ConcatGraphRepresentation(
            self,
            atom_indices,
        )


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
                "Graph data must contain `ptr` for selected-atom concatenation."
            )

        ptr = data["ptr"].to(
            device=features.device,
            dtype=torch.long,
        )
        atoms_per_graph = ptr[1:] - ptr[:-1]
        if torch.any(atoms_per_graph <= self.max_selected_atom_index):
            raise RuntimeError(
                "A selected atom index exceeds the number of atoms "
                "in at least one system."
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