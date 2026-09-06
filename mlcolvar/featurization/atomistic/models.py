from typing import Dict, Optional, Sequence, Tuple

import torch

from mlcolvar.core import BaseGNN, FeedForward

from .featurizer import AtomisticFeaturizer


__all__ = ["AtomisticModel"]


class _BaseAtomisticModel(BaseGNN):
    """Shared implementation for atomistic CV models."""

    def __init__(
        self,
        featurizer: AtomisticFeaturizer,
        readout_in_features: int,
        n_out: int,
        hidden_layers: Tuple[int, ...],
    ) -> None:
        if n_out <= 0:
            raise ValueError("`n_out` must be positive.")

        if any(size <= 0 for size in hidden_layers):
            raise ValueError(
                "`hidden_layers` must contain positive integers."
            )

        super().__init__(
            n_out=n_out,
            dataset_for_initialization=None,
            pooling_operation=None,
            cutoff=float(
                featurizer.cutoff.detach().cpu().item()
            ),
            buffer=float(
                featurizer.buffer.detach().cpu().item()
            ),
            long_range_cutoff=float(
                featurizer.long_range_cutoff
                .detach()
                .cpu()
                .item()
            ),
            atomic_numbers=(
                featurizer.atomic_numbers
                .detach()
                .cpu()
                .tolist()
            ),
        )

        # External atomistic backbones construct their own
        # neighborhood representations.
        self._modules.pop("_radial_embedding", None)

        self.featurizer = featurizer

        self.readout = FeedForward(
            layers=[
                readout_in_features,
                *hidden_layers,
                n_out,
            ]
        )

        parameter = next(self.readout.parameters())

        self.register_buffer(
            "_readout_dtype_reference",
            torch.zeros(
                (),
                device=parameter.device,
                dtype=parameter.dtype,
            ),
        )

    def _cast_features(
        self,
        features: torch.Tensor,
    ) -> torch.Tensor:
        """Convert backbone features to readout device and dtype."""

        return features.to(
            device=self._readout_dtype_reference.device,
            dtype=self._readout_dtype_reference.dtype,
        )


class _PooledAtomisticModel(_BaseAtomisticModel):
    """Pool atom features before applying the readout."""

    def __init__(
        self,
        featurizer: AtomisticFeaturizer,
        n_out: int,
        hidden_layers: Tuple[int, ...],
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
        features = self.featurizer(
            data,
            cell=cell,
        )

        features = self._cast_features(features)

        return self.readout(features)


class _NodewiseAtomisticModel(_BaseAtomisticModel):
    """Apply the readout to each atom before pooling."""

    def __init__(
        self,
        featurizer: AtomisticFeaturizer,
        n_out: int,
        hidden_layers: Tuple[int, ...],
    ) -> None:
        if featurizer.sample_kind != "atom":
            raise ValueError(
                "`mode='nodewise'` requires an atom-level backbone."
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
        features = self.featurizer.forward_features(
            data,
            cell=cell,
        )

        features = self._cast_features(features)

        outputs = self.readout(features)

        return self.featurizer.pool(
            outputs,
            data,
        )


class _ConcatAtomisticModel(_BaseAtomisticModel):
    """Concatenate selected atom features before the readout."""

    __constants__ = [
        "n_selected_atoms",
        "max_selected_atom_index",
    ]

    def __init__(
        self,
        featurizer: AtomisticFeaturizer,
        selected_atom_indices: Sequence[int],
        n_out: int,
        hidden_layers: Tuple[int, ...],
    ) -> None:
        if featurizer.sample_kind != "atom":
            raise ValueError(
                "`mode='concat'` requires an atom-level backbone."
            )

        indices = [
            int(index)
            for index in selected_atom_indices
        ]

        if not indices:
            raise ValueError(
                "`selected_atom_indices` cannot be empty."
            )

        if any(index < 0 for index in indices):
            raise ValueError(
                "`selected_atom_indices` must contain "
                "non-negative indices."
            )

        if len(indices) != len(set(indices)):
            raise ValueError(
                "`selected_atom_indices` must not contain duplicates."
            )

        self.n_selected_atoms = len(indices)
        self.max_selected_atom_index = max(indices)

        super().__init__(
            featurizer=featurizer,
            readout_in_features=(
                featurizer.out_features
                * self.n_selected_atoms
            ),
            n_out=n_out,
            hidden_layers=hidden_layers,
        )

        self.register_buffer(
            "selected_atom_indices",
            torch.tensor(
                indices,
                dtype=torch.long,
            ),
        )

    def forward(
        self,
        data: Dict[str, torch.Tensor],
        cell: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if "ptr" not in data:
            raise KeyError(
                "Graph data must contain `ptr` for "
                "selected-atom concatenation."
            )

        features = self.featurizer.forward_features(
            data,
            cell=cell,
        )

        features = self._cast_features(features)

        ptr = data["ptr"].to(
            device=features.device,
            dtype=torch.long,
        )

        selected_atom_indices = (
            self.selected_atom_indices.to(
                device=features.device,
                dtype=torch.long,
            )
        )

        atoms_per_graph = ptr[1:] - ptr[:-1]

        if torch.any(
            atoms_per_graph
            <= self.max_selected_atom_index
        ):
            raise RuntimeError(
                "A selected atom index exceeds the number "
                "of atoms in at least one system."
            )

        global_indices = (
            ptr[:-1].unsqueeze(1)
            + selected_atom_indices.unsqueeze(0)
        )

        selected_features = features.index_select(
            0,
            global_indices.reshape(-1),
        )

        concatenated = selected_features.reshape(
            ptr.numel() - 1,
            self.n_selected_atoms * features.size(-1),
        )

        return self.readout(concatenated)


def AtomisticModel(
    featurizer: AtomisticFeaturizer,
    mode: str = "pooled",
    selected_atom_indices: Optional[Sequence[int]] = None,
    n_out: int = 1,
    hidden_layers: Tuple[int, ...] = (30, 30),
) -> BaseGNN:
    """Create an atomistic CV model.

    Parameters
    ----------
    featurizer
        Pretrained atomistic feature extractor.
    mode
        Readout strategy: ``"pooled"``, ``"nodewise"``,
        or ``"concat"``.
    selected_atom_indices
        Atom indices used by ``mode="concat"``.
    n_out
        Number of output CVs.
    hidden_layers
        Hidden dimensions of the trainable readout.
    """

    mode = mode.lower()

    if mode == "pooled":
        return _PooledAtomisticModel(
            featurizer=featurizer,
            n_out=n_out,
            hidden_layers=hidden_layers,
        )

    if mode == "nodewise":
        return _NodewiseAtomisticModel(
            featurizer=featurizer,
            n_out=n_out,
            hidden_layers=hidden_layers,
        )

    if mode == "concat":
        if selected_atom_indices is None:
            raise ValueError(
                "`selected_atom_indices` is required "
                "when mode='concat'."
            )

        return _ConcatAtomisticModel(
            featurizer=featurizer,
            selected_atom_indices=selected_atom_indices,
            n_out=n_out,
            hidden_layers=hidden_layers,
        )

    raise ValueError(
        "`mode` must be 'pooled', 'nodewise', or 'concat'. "
        f"Found {mode!r}."
    )