from typing import Dict, Optional, Sequence, Tuple

import torch

from mlcolvar.core import BaseGNN, FeedForward

from .featurizer import AtomisticFeaturizer


__all__ = [
    "AtomisticPooledModel",
    "AtomisticNodewiseModel",
    "AtomisticConcatModel",
]


class _BaseAtomisticReadoutModel(BaseGNN):
    """Shared initialization for atomistic CV readout models."""

    def __init__(
        self,
        featurizer: AtomisticFeaturizer,
        readout_in_features: int,
        n_out: int,
        hidden_layers: Tuple[int, ...],
    ) -> None:
        if n_out <= 0:
            raise ValueError("`n_out` must be positive.")

        if readout_in_features <= 0:
            raise ValueError("`readout_in_features` must be positive.")

        if any(size <= 0 for size in hidden_layers):
            raise ValueError("`hidden_layers` must contain positive integers.")

        super().__init__(
            n_out=n_out,
            dataset_for_initialization=None,
            pooling_operation=None,
            cutoff=float(featurizer.cutoff.detach().cpu().item()),
            buffer=float(featurizer.buffer.detach().cpu().item()),
            long_range_cutoff=float(
                featurizer.long_range_cutoff.detach().cpu().item()
            ),
            atomic_numbers=featurizer.atomic_numbers.detach().cpu().tolist(),
        )

        # BaseGNN's radial embedding is not used by external backbones.
        self._modules.pop("_radial_embedding", None)

        self.featurizer = featurizer
        self.readout = FeedForward(
            layers=[
                readout_in_features,
                *hidden_layers,
                n_out,
            ],
        )

        readout_parameter = next(self.readout.parameters())
        self.register_buffer(
            "_readout_dtype_reference",
            torch.zeros(
                (),
                device=readout_parameter.device,
                dtype=readout_parameter.dtype,
            ),
        )

    def _cast_features(self, features: torch.Tensor) -> torch.Tensor:
        """Convert features to the readout device and precision."""

        return features.to(
            device=self._readout_dtype_reference.device,
            dtype=self._readout_dtype_reference.dtype,
        )


class AtomisticPooledModel(_BaseAtomisticReadoutModel):
    """Pool atomistic features before applying the CV readout.

    The data flow is:

    ``node features -> pooling -> CV readout``

    This is the simplest and least expensive aggregation strategy.
    """

    def __init__(
        self,
        featurizer: AtomisticFeaturizer,
        n_out: int = 1,
        hidden_layers: Tuple[int, ...] = (30, 30),
    ) -> None:
        super().__init__(
            featurizer=featurizer,
            readout_in_features=featurizer.out_features,
            n_out=n_out,
            hidden_layers=hidden_layers,
        )

    def forward(
        self,
        data: Dict[str, torch.Tensor],
        cell: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        system_features = self.featurizer(data, cell=cell)
        system_features = self._cast_features(system_features)
        return self.readout(system_features)


class AtomisticNodewiseModel(_BaseAtomisticReadoutModel):
    """Apply a shared CV readout to every node before pooling.

    The data flow is:

    ``node features -> shared node-wise readout -> pooling``

    Since the same readout is applied to every atom and the results are
    aggregated by mean or sum, this construction remains permutation
    invariant.
    """

    def __init__(
        self,
        featurizer: AtomisticFeaturizer,
        n_out: int = 1,
        hidden_layers: Tuple[int, ...] = (30, 30),
    ) -> None:
        if featurizer.sample_kind != "atom":
            raise ValueError(
                "`AtomisticNodewiseModel` requires an atom-level backbone."
            )

        super().__init__(
            featurizer=featurizer,
            readout_in_features=featurizer.out_features,
            n_out=n_out,
            hidden_layers=hidden_layers,
        )

    def forward(
        self,
        data: Dict[str, torch.Tensor],
        cell: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        node_features = self.featurizer.forward_features(data, cell=cell)
        node_features = self._cast_features(node_features)
        node_outputs = self.readout(node_features)

        return self.featurizer.pool_node_features(
            node_features=node_outputs,
            data=data,
        )


class AtomisticConcatModel(_BaseAtomisticReadoutModel):
    """Concatenate selected atom features before the CV readout.

    The data flow is:

    ``selected node features -> concatenation -> CV readout``

    This strategy preserves atom-resolved information, but is not
    permutation invariant. All systems must use the same atom ordering.

    Parameters
    ----------
    featurizer
        Atomistic featurizer returning atom-level features.
    selected_atom_indices
        Zero-based atom indices local to each system. The order of these
        indices determines the concatenation order.
    n_out
        Number of CV outputs.
    hidden_layers
        Hidden dimensions of the trainable CV readout.
    """

    __constants__ = [
        "n_selected_atoms",
        "max_selected_atom_index",
    ]

    def __init__(
        self,
        featurizer: AtomisticFeaturizer,
        selected_atom_indices: Sequence[int],
        n_out: int = 1,
        hidden_layers: Tuple[int, ...] = (30, 30),
    ) -> None:
        if featurizer.sample_kind != "atom":
            raise ValueError(
                "`AtomisticConcatModel` requires an atom-level backbone."
            )

        indices = [int(index) for index in selected_atom_indices]

        if len(indices) == 0:
            raise ValueError("`selected_atom_indices` cannot be empty.")

        if any(index < 0 for index in indices):
            raise ValueError(
                "`selected_atom_indices` must contain non-negative indices."
            )

        if len(set(indices)) != len(indices):
            raise ValueError(
                "`selected_atom_indices` must not contain duplicates."
            )

        n_selected_atoms = len(indices)
        max_selected_atom_index = max(indices)

        super().__init__(
            featurizer=featurizer,
            readout_in_features=featurizer.out_features * n_selected_atoms,
            n_out=n_out,
            hidden_layers=hidden_layers,
        )

        self.n_selected_atoms = n_selected_atoms
        self.max_selected_atom_index = max_selected_atom_index

        self.register_buffer(
            "selected_atom_indices",
            torch.tensor(indices, dtype=torch.long),
        )

    def forward(
        self,
        data: Dict[str, torch.Tensor],
        cell: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if "ptr" not in data:
            raise KeyError(
                "Graph data must contain `ptr` for selected-atom concatenation."
            )

        node_features = self.featurizer.forward_features(data, cell=cell)
        node_features = self._cast_features(node_features)

        ptr = data["ptr"].to(
            device=node_features.device,
            dtype=torch.long,
        )
        selected_atom_indices = self.selected_atom_indices.to(
            device=node_features.device,
            dtype=torch.long,
        )

        n_graphs = ptr.numel() - 1
        atoms_per_graph = ptr[1:] - ptr[:-1]

        if torch.any(atoms_per_graph <= self.max_selected_atom_index):
            raise RuntimeError(
                "A selected atom index exceeds the number of atoms "
                "in at least one system."
            )

        global_indices = (
            ptr[:-1].unsqueeze(1)
            + selected_atom_indices.unsqueeze(0)
        )
        selected_features = node_features.index_select(
            0,
            global_indices.reshape(-1),
        )
        concatenated_features = selected_features.reshape(
            n_graphs,
            self.n_selected_atoms * node_features.size(-1),
        )

        return self.readout(concatenated_features)
