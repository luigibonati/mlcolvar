from typing import Dict, Optional, Sequence, Tuple

import torch
from torch import nn

from mlcolvar.core import BaseGNN, FeedForward
from mlcolvar.integrations.utils import (
    align_node_attrs,
    infer_num_graphs,
)

__all__ = [
    "BaseAtomisticBackbone",
    "AtomisticFeaturizer",
    "AtomisticPooledModel",
    "AtomisticNodewiseModel",
    "AtomisticConcatModel",
]


_METADATA_BUFFERS = (
    "feature_dim",
    "atomic_numbers",
    "cutoff",
    "buffer",
    "long_range_cutoff",
)


def _copy_backbone_metadata(
    module: nn.Module,
    backbone: nn.Module,
) -> None:
    """Copy standardized backbone metadata to another module."""

    for name in _METADATA_BUFFERS:
        module.register_buffer(
            name,
            getattr(backbone, name).detach().clone(),
        )


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

        atomic_numbers = [
            int(number)
            for number in atomic_numbers
        ]

        if out_features <= 0:
            raise ValueError(
                "`out_features` must be positive."
            )

        if len(atomic_numbers) == 0:
            raise ValueError(
                "`atomic_numbers` cannot be empty."
            )

        if len(set(atomic_numbers)) != len(atomic_numbers):
            raise ValueError(
                "`atomic_numbers` must not contain duplicates."
            )

        if any(number <= 0 for number in atomic_numbers):
            raise ValueError(
                "`atomic_numbers` must contain positive integers."
            )

        if cutoff <= 0.0:
            raise ValueError(
                "`cutoff` must be positive."
            )

        if buffer < 0.0:
            raise ValueError(
                "`buffer` must be non-negative."
            )

        if (
            long_range_cutoff >= 0.0
            and long_range_cutoff <= cutoff
        ):
            raise ValueError(
                "`long_range_cutoff` must be negative or larger "
                "than `cutoff`."
            )

        if sample_kind not in ("atom", "system"):
            raise ValueError(
                "`sample_kind` must be either 'atom' or 'system'."
            )

        self.out_features = int(out_features)
        self.sample_kind = sample_kind
        self.full_neighbor_list = bool(
            full_neighbor_list
        )

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

    def forward(
        self,
        data: Dict[str, torch.Tensor],
        cell: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return atom- or system-level features."""

        raise NotImplementedError


class AtomisticFeaturizer(nn.Module):
    """Extract features from a pretrained atomistic backbone.

    The default :meth:`forward` method converts atom-level features into
    system-level features using mean or sum pooling.

    Raw atom-level features can be obtained through
    :meth:`forward_features`.
    """

    __constants__ = [
        "out_features",
        "sample_kind",
        "pooling",
        "freeze",
        "full_neighbor_list",
    ]

    def __init__(
        self,
        backbone: BaseAtomisticBackbone,
        pooling: str = "mean",
        freeze: bool = True,
    ) -> None:
        super().__init__()

        if pooling not in ("mean", "sum"):
            raise ValueError(
                "`pooling` must be either 'mean' or 'sum'."
            )

        self.backbone = backbone
        self.out_features = backbone.out_features
        self.sample_kind = backbone.sample_kind
        self.pooling = pooling
        self.freeze = bool(freeze)
        self.full_neighbor_list = (
            backbone.full_neighbor_list
        )

        _copy_backbone_metadata(
            self,
            backbone,
        )

        if self.freeze:
            for parameter in self.backbone.parameters():
                parameter.requires_grad_(False)

            self.backbone.eval()

    @torch.jit.unused
    def align_dataset(
        self,
        dataset,
    ):
        """Align dataset node attributes with the model element table."""

        return align_node_attrs(
            dataset=dataset,
            target_atomic_numbers=self.atomic_numbers,
        )

    def train(
        self,
        mode: bool = True,
    ):
        """Keep a frozen backbone in evaluation mode."""

        super().train(mode)

        if self.freeze:
            self.backbone.eval()

        return self

    def forward_features(
        self,
        data: Dict[str, torch.Tensor],
        cell: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return raw atom- or system-level backbone features."""

        return self.backbone(
            data,
            cell=cell,
        )

    def forward(
        self,
        data: Dict[str, torch.Tensor],
        cell: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return system-level features using the configured pooling."""

        features = self.forward_features(
            data,
            cell=cell,
        )

        if self.sample_kind == "system":
            return features

        return self.pool_node_features(
            node_features=features,
            data=data,
        )

    def pool_node_features(
        self,
        node_features: torch.Tensor,
        data: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Pool node-level values into system-level values.

        If ``system_masks`` is present, only nodes with non-zero mask
        values contribute to the pooling operation.
        """

        if "batch" not in data:
            raise KeyError(
                "Graph data must contain `batch` for node pooling."
            )

        batch = data["batch"].to(
            device=node_features.device,
            dtype=torch.long,
        )

        n_graphs = infer_num_graphs(data)

        if "system_masks" in data:
            mask = data["system_masks"].reshape(
                -1,
                1,
            ).to(
                device=node_features.device,
                dtype=node_features.dtype,
            )
        else:
            mask = node_features.new_ones(
                (
                    node_features.size(0),
                    1,
                )
            )

        output = node_features.new_zeros(
            (
                n_graphs,
                node_features.size(-1),
            )
        )

        output.index_add_(
            0,
            batch,
            node_features * mask,
        )

        if self.pooling == "sum":
            return output

        counts = node_features.new_zeros(
            (
                n_graphs,
                1,
            )
        )

        counts.index_add_(
            0,
            batch,
            mask,
        )

        return output / counts.clamp_min(1.0)


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
            raise ValueError(
                "`n_out` must be positive."
            )

        if readout_in_features <= 0:
            raise ValueError(
                "`readout_in_features` must be positive."
            )

        if any(
            size <= 0
            for size in hidden_layers
        ):
            raise ValueError(
                "`hidden_layers` must contain positive integers."
            )

        super().__init__(
            n_out=n_out,
            dataset_for_initialization=None,
            pooling_operation=None,
            cutoff=float(
                featurizer.cutoff
                .detach()
                .cpu()
                .item()
            ),
            buffer=float(
                featurizer.buffer
                .detach()
                .cpu()
                .item()
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

        # BaseGNN's radial embedding is not used by external backbones.
        self._modules.pop(
            "_radial_embedding",
            None,
        )

        self.featurizer = featurizer

        self.readout = FeedForward(
            layers=[
                readout_in_features,
                *hidden_layers,
                n_out,
            ],
        )

        readout_parameter = next(
            self.readout.parameters()
        )

        self.register_buffer(
            "_readout_dtype_reference",
            torch.zeros(
                (),
                device=readout_parameter.device,
                dtype=readout_parameter.dtype,
            ),
        )

    def _cast_features(
        self,
        features: torch.Tensor,
    ) -> torch.Tensor:
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
        system_features = self.featurizer(
            data,
            cell=cell,
        )

        system_features = self._cast_features(
            system_features
        )

        return self.readout(
            system_features
        )


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
                "`AtomisticNodewiseModel` requires an atom-level "
                "backbone."
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
        node_features = self.featurizer.forward_features(
            data,
            cell=cell,
        )

        node_features = self._cast_features(
            node_features
        )

        node_outputs = self.readout(
            node_features
        )

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
                "`AtomisticConcatModel` requires an atom-level "
                "backbone."
            )

        indices = [
            int(index)
            for index in selected_atom_indices
        ]

        if len(indices) == 0:
            raise ValueError(
                "`selected_atom_indices` cannot be empty."
            )

        if any(index < 0 for index in indices):
            raise ValueError(
                "`selected_atom_indices` must contain "
                "non-negative indices."
            )

        if len(set(indices)) != len(indices):
            raise ValueError(
                "`selected_atom_indices` must not contain "
                "duplicates."
            )

        n_selected_atoms = len(indices)
        max_selected_atom_index = max(indices)

        super().__init__(
            featurizer=featurizer,
            readout_in_features=(
                featurizer.out_features
                * n_selected_atoms
            ),
            n_out=n_out,
            hidden_layers=hidden_layers,
        )

        self.n_selected_atoms = n_selected_atoms
        self.max_selected_atom_index = (
            max_selected_atom_index
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
                "Graph data must contain `ptr` for selected-atom "
                "concatenation."
            )

        node_features = self.featurizer.forward_features(
            data,
            cell=cell,
        )

        node_features = self._cast_features(
            node_features
        )

        ptr = data["ptr"].to(
            device=node_features.device,
            dtype=torch.long,
        )

        selected_atom_indices = (
            self.selected_atom_indices.to(
                device=node_features.device,
                dtype=torch.long,
            )
        )

        n_graphs = ptr.numel() - 1

        # Number of atoms in each system.
        atoms_per_graph = ptr[1:] - ptr[:-1]

        if torch.any(
            atoms_per_graph
            <= self.max_selected_atom_index
        ):
            raise RuntimeError(
                "A selected atom index exceeds the number of atoms "
                "in at least one system."
            )

        # Convert local indices to global indices in the batched graph:
        #
        # ptr[:-1]                  [n_graphs]
        # selected_atom_indices     [n_selected_atoms]
        # global_indices            [n_graphs, n_selected_atoms]
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
            self.n_selected_atoms
            * node_features.size(-1),
        )

        return self.readout(
            concatenated_features
        )