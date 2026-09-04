from typing import Optional

import torch
from torch import nn
from torch_geometric.loader import DataLoader as GraphDataLoader

from mlcolvar.core.loss.utils.smart_derivatives import SmartDerivatives
from mlcolvar.data import DictDataset

from .featurizers import (
    _BaseGraphFeaturizer,
    _BaseTensorFeaturizer,
)


__all__ = [
    "CachedLatentDerivatives",
    "precompute_committor_cache",
]


class CachedLatentDerivatives(SmartDerivatives):
    """Map latent gradients dq/dh to coordinate gradients dq/dR."""

    def __init__(self, jacobian: torch.Tensor) -> None:
        nn.Module.__init__(self)
        self.register_buffer(
            "jacobian",
            jacobian,
            persistent=False,
        )

    def forward(
        self,
        gradient_latent: torch.Tensor,
        ref_idx: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if ref_idx is None:
            raise ValueError("`ref_idx` is required.")

        ref_idx = ref_idx.reshape(-1).to(
            device=self.jacobian.device,
            dtype=torch.long,
        )

        jacobian = self.jacobian.index_select(
            0,
            ref_idx,
        )

        return torch.einsum(
            "bl,b...l->b...",
            gradient_latent,
            jacobian,
        )


def _dataset_keys(
    dataset: DictDataset,
) -> tuple[str, ...]:
    keys = dataset.keys

    if callable(keys):
        keys = keys()

    return tuple(keys)


def _variational_indices(
    labels: torch.Tensor,
    separate_boundary_dataset: bool,
) -> torch.Tensor:
    if separate_boundary_dataset:
        indices = torch.nonzero(
            labels.reshape(-1) > 1,
            as_tuple=False,
        ).reshape(-1)
    else:
        indices = torch.arange(
            labels.numel(),
            device=labels.device,
        )

    if indices.numel() == 0:
        raise ValueError(
            "No variational samples were found."
        )

    return indices


def _precompute_tensor_committor_cache(
    featurizer: _BaseTensorFeaturizer,
    dataset: DictDataset,
    descriptor_derivatives: SmartDerivatives,
    batch_size: int = 1024,
    device: str | torch.device = "cuda",
    output_device: str | torch.device = "cpu",
    separate_boundary_dataset: bool = True,
) -> tuple[
    DictDataset,
    CachedLatentDerivatives,
]:
    """Cache tensor latent features and dh/dR for head-only training."""

    if not featurizer.freeze:
        raise RuntimeError(
            "Caching requires `featurizer.freeze=True`."
        )

    if batch_size <= 0:
        raise ValueError(
            "`batch_size` must be positive."
        )

    keys = _dataset_keys(dataset)

    required = (
        "data",
        "labels",
        "weights",
        "ref_idx",
    )

    missing = [
        key
        for key in required
        if key not in keys
    ]

    if missing:
        raise KeyError(
            f"Dataset is missing required keys: {missing}."
        )

    device = torch.device(device)
    output_device = torch.device(output_device)

    x = dataset["data"]
    labels = dataset["labels"].reshape(-1)
    ref_idx = dataset["ref_idx"].reshape(-1).long()

    if (
        len(x) != len(labels)
        or len(x) != len(ref_idx)
    ):
        raise ValueError(
            "`data`, `labels`, and `ref_idx` "
            "must have equal lengths."
        )

    featurizer = featurizer.to(device).eval()
    descriptor_derivatives = (
        descriptor_derivatives.to(device)
    )

    latent = featurizer.precompute(
        x,
        batch_size=batch_size,
        device=device,
        output_device=output_device,
    )

    latent_dim = latent.shape[-1]

    indices = _variational_indices(
        labels,
        separate_boundary_dataset,
    )

    n_reference = int(
        ref_idx.max().item()
    ) + 1

    latent_jacobian = None

    with torch.enable_grad():
        for start in range(
            0,
            indices.numel(),
            batch_size,
        ):
            batch_indices = indices[
                start : start + batch_size
            ]

            x_batch = (
                x.index_select(
                    0,
                    batch_indices.to(x.device),
                )
                .to(device)
                .detach()
                .requires_grad_(True)
            )

            ref_batch = ref_idx.index_select(
                0,
                batch_indices.to(ref_idx.device),
            ).to(device)

            h_batch = featurizer(
                x_batch
            ).reshape(
                x_batch.shape[0],
                latent_dim,
            )

            dh_dR_parts = []

            for latent_index in range(
                latent_dim
            ):
                dh_dx = torch.autograd.grad(
                    outputs=(
                        h_batch[
                            :,
                            latent_index,
                        ].sum()
                    ),
                    inputs=x_batch,
                    retain_graph=(
                        latent_index + 1
                        < latent_dim
                    ),
                    create_graph=False,
                )[0]

                dh_dR_parts.append(
                    descriptor_derivatives(
                        dh_dx,
                        ref_batch,
                    )
                )

            batch_jacobian = torch.stack(
                dh_dR_parts,
                dim=-1,
            )

            if latent_jacobian is None:
                latent_jacobian = torch.zeros(
                    (
                        n_reference,
                        *batch_jacobian.shape[1:],
                    ),
                    dtype=batch_jacobian.dtype,
                    device=output_device,
                )

            latent_jacobian.index_copy_(
                0,
                ref_batch.to(output_device),
                batch_jacobian.to(
                    output_device
                ),
            )

    if latent_jacobian is None:
        raise RuntimeError(
            "Failed to construct the latent Jacobian."
        )

    cached_data = {}

    for key in keys:
        value = dataset[key]

        cached_data[key] = (
            value.to(output_device)
            if torch.is_tensor(value)
            else value
        )

    cached_data["data"] = latent

    return (
        DictDataset(cached_data),
        CachedLatentDerivatives(
            latent_jacobian
        ),
    )


def _precompute_graph_committor_cache(
    featurizer: _BaseGraphFeaturizer,
    dataset: DictDataset,
    batch_size: int = 256,
    device: str | torch.device = "cuda",
    output_device: str | torch.device = "cpu",
    separate_boundary_dataset: bool = True,
) -> tuple[
    DictDataset,
    CachedLatentDerivatives,
]:
    """Cache graph-level latent features and dh/dR.

    The dense derivative representation requires every graph
    to contain the same number of atoms. This is suitable for
    fixed-composition molecular trajectories such as alanine
    dipeptide.
    """

    if not featurizer.freeze:
        raise RuntimeError(
            "Caching requires `featurizer.freeze=True`."
        )

    if batch_size <= 0:
        raise ValueError(
            "`batch_size` must be positive."
        )

    if (
        dataset.metadata.get("data_type")
        != "graphs"
    ):
        raise TypeError(
            "Expected a graph-based `DictDataset`."
        )

    device = torch.device(device)
    output_device = torch.device(output_device)

    featurizer = (
        featurizer.to(device).eval()
    )

    loader = GraphDataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    latent_parts = []
    label_parts = []
    weight_parts = []
    jacobian_parts = []

    expected_n_atoms = None

    with torch.enable_grad():
        for batch in loader:
            graph = batch[
                "data_list"
            ].to(device)

            n_graphs = int(
                graph.num_graphs
            )

            counts = torch.bincount(
                graph.batch,
                minlength=n_graphs,
            )

            if not torch.all(
                counts == counts[0]
            ):
                raise ValueError(
                    "Graph caching currently "
                    "requires a fixed atom count."
                )

            n_atoms = int(
                counts[0].item()
            )

            if expected_n_atoms is None:
                expected_n_atoms = n_atoms
            elif n_atoms != expected_n_atoms:
                raise ValueError(
                    "All graph batches must have "
                    "the same atom count."
                )

            graph.positions = (
                graph.positions
                .detach()
                .requires_grad_(True)
            )

            latent_batch = featurizer(
                graph
            ).reshape(
                n_graphs,
                featurizer.out_features,
            )

            labels_batch = (
                graph.graph_labels
                .reshape(-1)
            )

            weights_batch = (
                graph.weight
                .reshape(-1)
            )

            if separate_boundary_dataset:
                local_indices = torch.nonzero(
                    labels_batch > 1,
                    as_tuple=False,
                ).reshape(-1)
            else:
                local_indices = torch.arange(
                    n_graphs,
                    device=(
                        labels_batch.device
                    ),
                )

            # Allocate zeros for boundary rows when
            # boundaries are excluded.
            batch_jacobian = torch.zeros(
                (
                    n_graphs,
                    n_atoms,
                    graph.positions.shape[-1],
                    featurizer.out_features,
                ),
                dtype=graph.positions.dtype,
                device=device,
            )

            if local_indices.numel() > 0:
                latent_selected = (
                    latent_batch.index_select(
                        0,
                        local_indices,
                    )
                )

                gradients = []

                for latent_index in range(
                    featurizer.out_features
                ):
                    gradient_positions = (
                        torch.autograd.grad(
                            outputs=(
                                latent_selected[
                                    :,
                                    latent_index,
                                ].sum()
                            ),
                            inputs=(
                                graph.positions
                            ),
                            retain_graph=(
                                latent_index + 1
                                < featurizer.out_features
                            ),
                            create_graph=False,
                        )[0]
                    )

                    gradient_positions = (
                        gradient_positions.reshape(
                            n_graphs,
                            n_atoms,
                            graph.positions.shape[-1],
                        )
                    )

                    gradients.append(
                        gradient_positions.index_select(
                            0,
                            local_indices,
                        )
                    )

                selected_jacobian = torch.stack(
                    gradients,
                    dim=-1,
                )

                batch_jacobian.index_copy_(
                    0,
                    local_indices,
                    selected_jacobian,
                )

            latent_parts.append(
                latent_batch
                .detach()
                .to(output_device)
            )

            label_parts.append(
                labels_batch
                .detach()
                .to(output_device)
            )

            weight_parts.append(
                weights_batch
                .detach()
                .to(output_device)
            )

            jacobian_parts.append(
                batch_jacobian
                .detach()
                .to(output_device)
            )

    if not latent_parts:
        raise ValueError(
            "The graph dataset is empty."
        )

    latent = torch.cat(
        latent_parts,
        dim=0,
    )

    labels = torch.cat(
        label_parts,
        dim=0,
    )

    weights = torch.cat(
        weight_parts,
        dim=0,
    )

    latent_jacobian = torch.cat(
        jacobian_parts,
        dim=0,
    )

    ref_idx = torch.arange(
        len(latent),
        dtype=torch.long,
        device=output_device,
    )

    cached_dataset = DictDataset(
        {
            "data": latent,
            "labels": labels,
            "weights": weights,
            "ref_idx": ref_idx,
        }
    )

    return (
        cached_dataset,
        CachedLatentDerivatives(
            latent_jacobian
        ),
    )


def precompute_committor_cache(
    featurizer: nn.Module,
    dataset: DictDataset,
    descriptor_derivatives: Optional[
        SmartDerivatives
    ] = None,
    batch_size: Optional[int] = None,
    device: str | torch.device = "cuda",
    output_device: str | torch.device = "cpu",
    separate_boundary_dataset: bool = True,
) -> tuple[
    DictDataset,
    CachedLatentDerivatives,
]:
    """Precompute latent features and coordinate Jacobians.

    The tensor or graph implementation is selected automatically
    from the transfer featurizer type.

    Parameters
    ----------
    featurizer
        Frozen featurizer created with ``TransferFeaturizer``.
    dataset
        Dataset used for committor training.
    descriptor_derivatives
        Mapping from descriptor gradients to coordinate gradients.
        Required for tensor featurizers and unused for graph
        featurizers.
    batch_size
        Number of samples or graphs processed per batch. Defaults
        to 1024 for tensor models and 256 for graph models.
    device
        Device used for feature and Jacobian computation.
    output_device
        Device on which cached tensors are stored.
    separate_boundary_dataset
        Whether labels <= 1 correspond to boundary-state samples
        and should be excluded from variational Jacobian
        computation.
    """

    if isinstance(
        featurizer,
        _BaseTensorFeaturizer,
    ):
        if descriptor_derivatives is None:
            raise ValueError(
                "`descriptor_derivatives` is "
                "required for tensor-based "
                "committor caching."
            )

        return _precompute_tensor_committor_cache(
            featurizer=featurizer,
            dataset=dataset,
            descriptor_derivatives=(
                descriptor_derivatives
            ),
            batch_size=(
                1024
                if batch_size is None
                else batch_size
            ),
            device=device,
            output_device=output_device,
            separate_boundary_dataset=(
                separate_boundary_dataset
            ),
        )

    if isinstance(
        featurizer,
        _BaseGraphFeaturizer,
    ):
        if descriptor_derivatives is not None:
            raise ValueError(
                "`descriptor_derivatives` should "
                "not be provided for graph-based "
                "committor caching."
            )

        return _precompute_graph_committor_cache(
            featurizer=featurizer,
            dataset=dataset,
            batch_size=(
                256
                if batch_size is None
                else batch_size
            ),
            device=device,
            output_device=output_device,
            separate_boundary_dataset=(
                separate_boundary_dataset
            ),
        )

    raise TypeError(
        "`featurizer` must be created with "
        "`TransferFeaturizer`. "
        f"Found {type(featurizer)}."
    )
