"""Common interfaces for pretrained atomistic models."""

from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch import nn

from mlcolvar.core import BaseGNN, FeedForward


__all__ = [
    "align_node_attrs",
    "BaseAtomisticBackbone",
    "AtomisticFeaturizer",
    "AtomisticModel",
]


class BaseAtomisticBackbone(nn.Module):
    """Base class for pretrained atomistic-model backbones.

    A model-specific backbone converts the native output of an external
    atomistic model into one canonical representation.

    Implementations must return either:

    - atom-level features with shape ``[n_atoms, out_features]``;
    - system-level features with shape ``[n_graphs, out_features]``.

    The output kind is specified by ``sample_kind``.

    Parameters
    ----------
    out_features
        Number of features returned for each atom or system.
    atomic_numbers
        Atomic numbers supported by the external model. Their order must
        match the species encoding used to construct ``node_attrs``.
    cutoff
        Short-range cutoff required by the external model.
    sample_kind
        Whether the backbone returns ``"atom"`` or ``"system"`` features.
    buffer
        Additional environment buffer used when constructing graphs.
    long_range_cutoff
        Optional cutoff for long-range edges. A negative value disables
        long-range edges.
    full_neighbor_list
        Whether the external model requires a full directed neighbor list.
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
                "`out_features` must be positive, "
                f"found {out_features}."
            )

        if len(atomic_numbers) == 0:
            raise ValueError(
                "`atomic_numbers` cannot be empty."
            )

        if len(set(atomic_numbers)) != len(atomic_numbers):
            raise ValueError(
                "`atomic_numbers` must not contain duplicates: "
                f"{atomic_numbers}."
            )

        if any(number <= 0 for number in atomic_numbers):
            raise ValueError(
                "`atomic_numbers` must contain positive integers: "
                f"{atomic_numbers}."
            )

        if cutoff <= 0.0:
            raise ValueError(
                "`cutoff` must be positive, "
                f"found {cutoff}."
            )

        if buffer < 0.0:
            raise ValueError(
                "`buffer` must be non-negative, "
                f"found {buffer}."
            )

        if (
            long_range_cutoff >= 0.0
            and long_range_cutoff <= cutoff
        ):
            raise ValueError(
                "`long_range_cutoff` must be negative or larger than "
                f"`cutoff`. Found cutoff={cutoff} and "
                f"long_range_cutoff={long_range_cutoff}."
            )

        if sample_kind not in ("atom", "system"):
            raise ValueError(
                "`sample_kind` must be either 'atom' or 'system', "
                f"found {sample_kind!r}."
            )

        self.out_features = int(out_features)
        self.sample_kind = sample_kind
        self.full_neighbor_list = bool(full_neighbor_list)

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
        """Atomistic backbones receive graphs rather than flat tensors."""
        return None

    def forward(
        self,
        data: Dict[str, torch.Tensor],
        cell: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute atom-level or system-level features.

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
            Features with shape ``[n_samples, out_features]``.
        """
        raise NotImplementedError
    
    def _get_ptr(
        self,
        data: Dict[str, torch.Tensor],
        n_atoms: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Return system boundaries in a batched graph."""

        if "ptr" in data:
            return data["ptr"].to(
                device=device,
                dtype=torch.long,
            )

        if "batch" in data:
            batch = data["batch"].to(
                device=device,
                dtype=torch.long,
            )

            if batch.numel() == 0:
                return torch.zeros(
                    1,
                    device=device,
                    dtype=torch.long,
                )

            n_systems = (
                int(
                    batch.max().item()
                )
                + 1
            )

            counts = torch.bincount(
                batch,
                minlength=n_systems,
            )

            return torch.cat(
                [
                    torch.zeros(
                        1,
                        device=device,
                        dtype=torch.long,
                    ),
                    counts.cumsum(
                        dim=0,
                    ),
                ],
                dim=0,
            )

        return torch.tensor(
            [
                0,
                n_atoms,
            ],
            device=device,
            dtype=torch.long,
        )

    def _prepare_cells(
        self,
        data: Dict[str, torch.Tensor],
        cell: Optional[torch.Tensor],
        n_systems: int,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """Normalize cells to shape ``[n_systems, 3, 3]``."""

        if cell is not None:
            cells = cell

        elif "cell" in data:
            cells = data["cell"]

        else:
            cells = positions.new_zeros(
                (
                    n_systems,
                    3,
                    3,
                )
            )

        cells = cells.to(
            device=positions.device,
            dtype=positions.dtype,
        )

        if cells.dim() == 2:
            if (
                cells.size(0) == 3
                and cells.size(1) == 3
                and n_systems == 1
            ):
                cells = cells.unsqueeze(
                    0
                )

            elif (
                cells.size(0)
                == 3 * n_systems
                and cells.size(1) == 3
            ):
                cells = cells.reshape(
                    n_systems,
                    3,
                    3,
                )

        if (
            cells.dim() != 3
            or cells.size(0) != n_systems
            or cells.size(1) != 3
            or cells.size(2) != 3
        ):
            raise ValueError(
                "Expected cell shape [3, 3], "
                "[n_systems, 3, 3], or "
                "[3 * n_systems, 3], but found "
                f"{tuple(cells.shape)}."
            )

        return cells

    def _prepare_pbc(
        self,
        data: Dict[str, torch.Tensor],
        cells: torch.Tensor,
        n_systems: int,
    ) -> torch.Tensor:
        """Return PBC flags with shape ``[n_systems, 3]``."""

        if "pbc" not in data:
            return (
                torch.linalg.vector_norm(
                    cells,
                    dim=-1,
                )
                > 0.0
            )

        pbc = data["pbc"].to(
            device=cells.device,
            dtype=torch.bool,
        )

        if pbc.dim() == 1:
            if pbc.numel() == 3:
                pbc = pbc.reshape(
                    1,
                    3,
                ).expand(
                    n_systems,
                    3,
                )

            elif pbc.numel() == (
                3 * n_systems
            ):
                pbc = pbc.reshape(
                    n_systems,
                    3,
                )

        if (
            pbc.dim() != 2
            or pbc.size(0) != n_systems
            or pbc.size(1) != 3
        ):
            raise ValueError(
                "Expected PBC shape [3] or "
                "[n_systems, 3], but found "
                f"{tuple(pbc.shape)}."
            )

        return pbc


class AtomisticFeaturizer(nn.Module):
    """Convert a pretrained atomistic backbone into graph-level features.

    This class provides the common functionality shared by all external
    atomistic integrations:

    - parameter freezing;
    - output validation;
    - atom-to-system pooling;
    - support for ``system_masks``;
    - standardized graph-level output.

    Parameters
    ----------
    backbone
        Model-specific atomistic backbone.
    pooling
        Operation used to convert atom-level features into graph-level
        features. Available options are ``"mean"`` and ``"sum"``.
        This option is ignored for system-level backbones.
    freeze
        Whether to freeze the pretrained backbone parameters.
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
                "`pooling` must be either 'mean' or 'sum', "
                f"found {pooling!r}."
            )

        self.backbone = backbone

        self.out_features = backbone.out_features
        self.sample_kind = backbone.sample_kind
        self.pooling = pooling
        self.freeze = bool(freeze)
        self.full_neighbor_list = backbone.full_neighbor_list

        # Expose the same serialized metadata as the wrapped backbone.
        self.register_buffer(
            "feature_dim",
            backbone.feature_dim.detach().clone(),
        )

        self.register_buffer(
            "atomic_numbers",
            backbone.atomic_numbers.detach().clone(),
        )

        self.register_buffer(
            "cutoff",
            backbone.cutoff.detach().clone(),
        )

        self.register_buffer(
            "buffer",
            backbone.buffer.detach().clone(),
        )

        self.register_buffer(
            "long_range_cutoff",
            backbone.long_range_cutoff.detach().clone(),
        )

        if self.freeze:
            for parameter in self.backbone.parameters():
                parameter.requires_grad_(False)

            self.backbone.eval()
        else:
            self.backbone.train(self.training)
            
    @torch.jit.unused
    def align_dataset(
        self,
        dataset,
    ):
        """Align a graph dataset with the backbone element table.

        The dataset node attributes are rebuilt using the atomic-number
        ordering required by the wrapped pretrained backbone.

        Parameters
        ----------
        dataset
            mlcolvar graph dataset containing ``data_list`` and
            ``metadata["atomic_numbers"]``.

        Returns
        -------
        dataset
            Dataset with aligned ``node_attrs``.
        """
        return align_node_attrs(
            dataset=dataset,
            target_atomic_numbers=self.atomic_numbers,
        )

    @property
    def in_features(self) -> Optional[int]:
        """Atomistic featurizers receive graphs rather than flat tensors."""
        return None

    def train(
        self,
        mode: bool = True,
    ):
        """Set training mode while keeping a frozen backbone in eval mode."""
        super().train(mode)

        if self.freeze:
            self.backbone.eval()

        return self

    def forward(
        self,
        data: Dict[str, torch.Tensor],
        cell: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute graph-level features."""

        features = self.backbone(
            data,
            cell=cell,
        )

        if (
            not torch.jit.is_scripting()
            and not torch.jit.is_tracing()
        ):
            self._validate_features(
                features=features,
                data=data,
            )

        if self.sample_kind == "system":
            return features

        return self._pool_node_features(
            node_features=features,
            data=data,
        )

    @torch.jit.unused
    def _validate_features(
        self,
        features: torch.Tensor,
        data: Dict[str, torch.Tensor],
    ) -> None:
        """Validate the standardized backbone output."""

        if not isinstance(features, torch.Tensor):
            raise RuntimeError(
                "The atomistic backbone must return a torch.Tensor."
            )

        if features.dim() != 2:
            raise RuntimeError(
                "The atomistic backbone must return a rank-2 tensor "
                "with shape [n_samples, out_features]."
            )

        if features.size(1) != self.out_features:
            raise RuntimeError(
                "Unexpected atomistic feature dimension: expected "
                f"{self.out_features}, found {features.size(1)}."
            )

        if self.sample_kind == "atom":
            if "batch" not in data:
                raise KeyError(
                    "Graph data must contain `batch` for atom-level "
                    "features."
                )

            if features.size(0) != data["batch"].size(0):
                raise RuntimeError(
                    "The number of atom-level features does not match "
                    "the number of graph nodes."
                )

        else:
            n_graphs = self._get_num_graphs(data)

            if features.size(0) != n_graphs:
                raise RuntimeError(
                    "The number of system-level features does not match "
                    f"the graph batch size. Expected {n_graphs}, found "
                    f"{features.size(0)}."
                )

    def _pool_node_features(
        self,
        node_features: torch.Tensor,
        data: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Pool node-level features into graph-level features."""

        if "batch" not in data:
            raise KeyError(
                "Graph data must contain `batch` for atom-level pooling."
            )

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
            "Cannot infer the number of graphs. Graph data must "
            "contain `ptr` or `n_system`."
        )


class AtomisticModel(BaseGNN):
    """Atomistic featurizer followed by a trainable CV readout.

    This generic wrapper allows any external pretrained atomistic backbone
    to be used by mlcolvar graph-based CV classes.

    Parameters
    ----------
    featurizer
        Atomistic featurizer returning graph-level features.
    n_out
        Number of output collective variables.
    hidden_layers
        Hidden dimensions of the trainable feed-forward readout.
    """

    def __init__(
        self,
        featurizer: AtomisticFeaturizer,
        n_out: int = 1,
        hidden_layers: Tuple[int, ...] = (30, 30),
    ) -> None:
        if n_out <= 0:
            raise ValueError(
                "`n_out` must be positive, "
                f"found {n_out}."
            )

        if any(size <= 0 for size in hidden_layers):
            raise ValueError(
                "`hidden_layers` must contain positive integers, "
                f"found {hidden_layers}."
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

        # AtomisticFeaturizer already performs node-to-graph pooling.
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

        # External backbones may use a fixed internal precision.
        # Always match the graph-level features to the trainable readout.
        readout_parameter = next(
            self.readout.parameters()
        )

        features = features.to(
            device=readout_parameter.device,
            dtype=readout_parameter.dtype,
        )

        return self.readout(
            features
        )
        

def align_node_attrs(
    dataset,
    target_atomic_numbers,
):
    """Align graph node attributes with an external model element table.

    Graph datasets typically encode atomic species as one-hot node
    attributes. The meaning of each column is determined by the ordering
    of ``dataset.metadata["atomic_numbers"]``.

    This function rebuilds all ``node_attrs`` tensors using a target
    atomic-number table, for example the element table of a pretrained
    MACE, PET, or NequIP model.

    The dataset is modified in place and returned for convenience.

    Parameters
    ----------
    dataset
        mlcolvar graph dataset containing ``data_list`` and metadata with
        an ``atomic_numbers`` entry.
    target_atomic_numbers
        Atomic-number table required by the external model. It can be a
        sequence of integers or a tensor.

    Returns
    -------
    dataset
        The input dataset with aligned ``node_attrs`` and updated metadata.
    """

    if not hasattr(dataset, "metadata"):
        raise TypeError(
            "The dataset must expose a `metadata` attribute."
        )

    if "atomic_numbers" not in dataset.metadata:
        raise KeyError(
            "The dataset metadata must contain `atomic_numbers`."
        )

    try:
        data_list = dataset["data_list"]
    except (KeyError, TypeError) as exc:
        raise KeyError(
            "The dataset must contain `data_list`."
        ) from exc

    source_atomic_numbers = torch.as_tensor(
        dataset.metadata["atomic_numbers"],
        dtype=torch.long,
    ).detach().cpu().reshape(-1)

    target_atomic_numbers = torch.as_tensor(
        target_atomic_numbers,
        dtype=torch.long,
    ).detach().cpu().reshape(-1)

    if source_atomic_numbers.numel() == 0:
        raise ValueError(
            "The source atomic-number table cannot be empty."
        )

    if target_atomic_numbers.numel() == 0:
        raise ValueError(
            "The target atomic-number table cannot be empty."
        )

    if torch.any(source_atomic_numbers <= 0):
        raise ValueError(
            "The source atomic-number table must contain "
            "positive integers."
        )

    if torch.any(target_atomic_numbers <= 0):
        raise ValueError(
            "The target atomic-number table must contain "
            "positive integers."
        )

    source_values = [
        int(number)
        for number in source_atomic_numbers.tolist()
    ]

    target_values = [
        int(number)
        for number in target_atomic_numbers.tolist()
    ]
    
    if source_values == target_values:
        return dataset

    if len(set(source_values)) != len(source_values):
        raise ValueError(
            "The source atomic-number table contains duplicates: "
            f"{source_values}."
        )

    if len(set(target_values)) != len(target_values):
        raise ValueError(
            "The target atomic-number table contains duplicates: "
            f"{target_values}."
        )

    target_indices = {
        atomic_number: index
        for index, atomic_number in enumerate(target_values)
    }

    missing = [
        atomic_number
        for atomic_number in source_values
        if atomic_number not in target_indices
    ]

    if missing:
        raise ValueError(
            "The target model does not support atomic numbers "
            f"{missing}."
        )

    # Map each old species column to the corresponding target column.
    source_to_target = torch.tensor(
        [
            target_indices[atomic_number]
            for atomic_number in source_values
        ],
        dtype=torch.long,
    )

    for graph_index, graph in enumerate(data_list):
        if "node_attrs" not in graph:
            raise KeyError(
                f"Graph {graph_index} does not contain `node_attrs`."
            )

        old_node_attrs = graph["node_attrs"]

        if old_node_attrs.dim() != 2:
            raise ValueError(
                f"Graph {graph_index} `node_attrs` must be rank 2, "
                f"found shape {tuple(old_node_attrs.shape)}."
            )

        if (
            old_node_attrs.size(1)
            != source_atomic_numbers.numel()
        ):
            raise ValueError(
                f"Graph {graph_index} contains "
                f"{old_node_attrs.size(1)} node-attribute columns, "
                "but the dataset metadata contains "
                f"{source_atomic_numbers.numel()} atomic numbers."
            )

        device = old_node_attrs.device

        # Recover the species index according to the old element table.
        local_species = old_node_attrs.argmax(
            dim=-1
        )

        # Map old species indices to target species indices.
        target_species = source_to_target.to(
            device=device
        )[local_species]

        # Rebuild the one-hot representation with the target width/order.
        new_node_attrs = old_node_attrs.new_zeros(
            (
                old_node_attrs.size(0),
                target_atomic_numbers.numel(),
            )
        )

        new_node_attrs.scatter_(
            dim=1,
            index=target_species.reshape(-1, 1),
            value=1,
        )

        graph["node_attrs"] = new_node_attrs

    dataset.metadata["atomic_numbers"] = target_values

    return dataset