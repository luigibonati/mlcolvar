from typing import Dict, Optional, Sequence, Tuple

import torch

from mlcolvar.core import BaseGNN, FeedForward

from .featurizer import AtomisticFeaturizer


__all__ = ["AtomisticModel"]


class AtomisticModel(BaseGNN):
    """CV model using features from a pretrained atomistic model.

    Parameters
    ----------
    featurizer
        Pretrained atomistic feature extractor.
    mode
        Readout strategy:

        - ``"pooled"``: pool atom features before the readout.
        - ``"nodewise"``: apply the readout per atom, then pool.
        - ``"concat"``: concatenate selected atom features before
          the readout.

    selected_atom_indices
        Atom indices used by ``mode="concat"``.
    n_out
        Number of output CVs.
    hidden_layers
        Hidden dimensions of the trainable readout.
    """

    __constants__ = [
        "mode",
        "n_selected_atoms",
        "max_selected_atom_index",
    ]

    def __init__(
        self,
        featurizer: AtomisticFeaturizer,
        mode: str = "pooled",
        selected_atom_indices: Optional[Sequence[int]] = None,
        n_out: int = 1,
        hidden_layers: Tuple[int, ...] = (30, 30),
    ) -> None:
        mode = mode.lower()

        if mode not in {"pooled", "nodewise", "concat"}:
            raise ValueError(
                "`mode` must be 'pooled', 'nodewise', or 'concat'. "
                f"Found {mode!r}."
            )

        if n_out <= 0:
            raise ValueError("`n_out` must be positive.")

        if any(size <= 0 for size in hidden_layers):
            raise ValueError(
                "`hidden_layers` must contain positive integers."
            )

        if mode in {"nodewise", "concat"}:
            if featurizer.sample_kind != "atom":
                raise ValueError(
                    f"`mode='{mode}'` requires an atom-level backbone."
                )

        indices = []

        if mode == "concat":
            if selected_atom_indices is None:
                raise ValueError(
                    "`selected_atom_indices` is required "
                    "when mode='concat'."
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

        n_selected_atoms = len(indices)
        max_selected_atom_index = (
            max(indices)
            if indices
            else -1
        )

        readout_in_features = featurizer.out_features

        if mode == "concat":
            readout_in_features *= n_selected_atoms

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

        # BaseGNN is kept for compatibility with the existing graph
        # interface and deployment metadata. External atomistic
        # backbones construct their own neighborhood representations.
        self._modules.pop("_radial_embedding", None)

        self.featurizer = featurizer
        self.mode = mode

        self.n_selected_atoms = n_selected_atoms
        self.max_selected_atom_index = max_selected_atom_index

        self.register_buffer(
            "selected_atom_indices",
            torch.tensor(
                indices,
                dtype=torch.long,
            ),
        )

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

    def _concat_selected_features(
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
            atoms_per_graph
            <= self.max_selected_atom_index
        ):
            raise RuntimeError(
                "A selected atom index exceeds the number "
                "of atoms in at least one system."
            )

        selected_atom_indices = self.selected_atom_indices.to(
            device=features.device,
        )

        global_indices = (
            ptr[:-1].unsqueeze(1)
            + selected_atom_indices.unsqueeze(0)
        )

        selected_features = features.index_select(
            0,
            global_indices.reshape(-1),
        )

        return selected_features.reshape(
            ptr.numel() - 1,
            self.n_selected_atoms * features.size(-1),
        )

    def forward(
        self,
        data: Dict[str, torch.Tensor],
        cell: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.mode == "pooled":
            features = self.featurizer(
                data,
                cell=cell,
            )

            return self.readout(
                self._cast_features(features)
            )

        features = self.featurizer.forward_features(
            data,
            cell=cell,
        )

        features = self._cast_features(features)

        if self.mode == "nodewise":
            outputs = self.readout(features)

            return self.featurizer.pool(
                outputs,
                data,
            )

        features = self._concat_selected_features(
            features,
            data,
        )

        return self.readout(features)