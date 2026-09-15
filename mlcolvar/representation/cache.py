from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
from torch_geometric.loader import DataLoader as GraphDataLoader

from mlcolvar.core.loss.utils.smart_derivatives import SmartDerivatives
from mlcolvar.data import DictDataset

from .base import GraphRepresentation, Representation, TensorRepresentation


__all__ = [
    "RepresentationCache",
    "IdentityDescriptorDerivatives",
    "CachedRepresentationDerivatives",
    "precompute_representation_cache",
    "precompute_committor_cache",
]


@dataclass
class RepresentationCache:
    """Task-independent cached representation and optional coordinate Jacobian."""

    features: torch.Tensor
    jacobian: Optional[torch.Tensor] = None
    reference_indices: Optional[torch.Tensor] = None
    

class IdentityDescriptorDerivatives(SmartDerivatives):
    """Treat descriptor components as coordinates of one particle."""

    def __init__(self) -> None:
        nn.Module.__init__(self)

    def forward(
        self,
        gradient_descriptor: torch.Tensor,
        ref_idx: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        del ref_idx
        return gradient_descriptor.unsqueeze(1)


class CachedRepresentationDerivatives(SmartDerivatives):
    """Apply the chain rule through a cached ``dh/dR`` tensor."""

    def __init__(self, jacobian: torch.Tensor) -> None:
        nn.Module.__init__(self)
        if jacobian.ndim < 2 or jacobian.shape[0] == 0:
            raise ValueError("`jacobian` must be a non-empty tensor.")
        self.register_buffer("jacobian", jacobian, persistent=False)

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
        if torch.any(ref_idx < 0):
            raise ValueError("`ref_idx` contains samples without cached derivatives.")
        if torch.any(ref_idx >= self.jacobian.shape[0]):
            raise IndexError("`ref_idx` exceeds the cached Jacobian size.")

        jacobian = self.jacobian.index_select(0, ref_idx)
        gradient_latent = gradient_latent.to(
            device=jacobian.device,
            dtype=jacobian.dtype,
        )
        return torch.einsum("bl,b...l->b...", gradient_latent, jacobian)


def _dataset_keys(dataset: DictDataset) -> tuple[str, ...]:
    keys = dataset.keys
    return tuple(keys() if callable(keys) else keys)


def _resolve_indices(
    n_samples: int,
    indices: Optional[torch.Tensor],
) -> torch.Tensor:
    if n_samples <= 0:
        raise ValueError("Cannot cache an empty dataset.")

    if indices is None:
        return torch.arange(n_samples, dtype=torch.long)

    indices = torch.as_tensor(indices).detach().cpu()
    if indices.dtype == torch.bool:
        if indices.numel() != n_samples:
            raise ValueError("Boolean indices must contain one value per sample.")
        indices = torch.nonzero(indices.reshape(-1), as_tuple=False).reshape(-1)
    else:
        indices = indices.reshape(-1).to(dtype=torch.long)

    if indices.numel() == 0:
        raise ValueError("No samples were selected for Jacobian caching.")
    if torch.any(indices < 0) or torch.any(indices >= n_samples):
        raise IndexError("`jacobian_indices` contains an out-of-range sample index.")

    return torch.unique(indices, sorted=True)


def _reference_map(
    n_samples: int,
    selected_indices: torch.Tensor,
    output_device: torch.device,
) -> torch.Tensor:
    refs = torch.full(
        (n_samples,),
        -1,
        dtype=torch.long,
        device=output_device,
    )
    selected_indices = selected_indices.to(output_device)
    refs.index_copy_(
        0,
        selected_indices,
        torch.arange(selected_indices.numel(), device=output_device),
    )
    return refs


def _representation_device(representation: Representation) -> torch.device:
    for tensor in representation.parameters():
        return tensor.device
    for tensor in representation.buffers():
        return tensor.device
    return torch.device("cpu")


def _cache_tensor_representation(
    representation: TensorRepresentation,
    dataset: DictDataset,
    *,
    descriptor_derivatives: Optional[SmartDerivatives],
    jacobian_indices: Optional[torch.Tensor],
    source_ref_idx: Optional[torch.Tensor],
    batch_size: int,
    device: Optional[str | torch.device],
    output_device: str | torch.device,
    compute_jacobian: bool,
) -> RepresentationCache:
    if not representation.freeze:
        raise RuntimeError("Caching requires `representation.freeze=True`.")
    if batch_size <= 0:
        raise ValueError("`batch_size` must be positive.")

    keys = _dataset_keys(dataset)
    if "data" not in keys:
        raise KeyError("Tensor representation caching requires `data`.")

    x = dataset["data"]
    n_samples = len(x)
    if n_samples == 0:
        raise ValueError("Cannot cache an empty dataset.")

    cell = dataset["cell"] if "cell" in keys else None
    original_device = _representation_device(representation)
    target_device = torch.device(device or original_device)
    output_device = torch.device(output_device)
    was_training = representation.training

    representation.to(target_device).eval()
    if descriptor_derivatives is not None:
        descriptor_derivatives.to(target_device)

    try:
        feature_parts = []
        with torch.no_grad():
            for start in range(0, n_samples, batch_size):
                stop = min(start + batch_size, n_samples)
                x_batch = x[start:stop].to(target_device)
                cell_batch = None if cell is None else cell[start:stop].to(target_device)
                feature_parts.append(
                    representation(x_batch, cell=cell_batch).detach().to(output_device)
                )
        features = torch.cat(feature_parts, dim=0)

        if not compute_jacobian:
            return RepresentationCache(features=features)

        if descriptor_derivatives is None:
            raise ValueError(
                "`descriptor_derivatives` is required for tensor coordinate Jacobians."
            )

        selected = _resolve_indices(n_samples, jacobian_indices)
        if source_ref_idx is None:
            source_ref_idx = torch.arange(n_samples, dtype=torch.long)
        else:
            source_ref_idx = torch.as_tensor(source_ref_idx).reshape(-1).long().cpu()
            if source_ref_idx.numel() != n_samples:
                raise ValueError("`source_ref_idx` must contain one value per sample.")

        jacobian_parts = []
        with torch.enable_grad():
            for start in range(0, selected.numel(), batch_size):
                idx = selected[start : start + batch_size]
                x_batch = (
                    x.index_select(0, idx.to(x.device))
                    .to(target_device)
                    .detach()
                    .requires_grad_(True)
                )
                cell_batch = (
                    None
                    if cell is None
                    else cell.index_select(0, idx.to(cell.device)).to(target_device)
                )
                source_ref = source_ref_idx.index_select(0, idx).to(target_device)
                if torch.any(source_ref < 0):
                    raise ValueError(
                        "A sample selected for Jacobian caching has negative `source_ref_idx`."
                    )

                h = representation(x_batch, cell=cell_batch).reshape(
                    x_batch.shape[0], representation.out_features
                )
                dh_dR = []
                for latent_index in range(representation.out_features):
                    dh_dx = torch.autograd.grad(
                        h[:, latent_index].sum(),
                        x_batch,
                        retain_graph=latent_index + 1 < representation.out_features,
                        create_graph=False,
                    )[0]
                    dh_dR.append(descriptor_derivatives(dh_dx, source_ref))

                jacobian_parts.append(
                    torch.stack(dh_dR, dim=-1).detach().to(output_device)
                )

        jacobian = torch.cat(jacobian_parts, dim=0)
        return RepresentationCache(
            features=features,
            jacobian=jacobian,
            reference_indices=_reference_map(n_samples, selected, output_device),
        )
    finally:
        representation.to(original_device)
        representation.train(was_training)


def _selected_local_indices(
    selected: torch.Tensor,
    offset: int,
    n_graphs: int,
) -> torch.Tensor:
    mask = (selected >= offset) & (selected < offset + n_graphs)
    return selected[mask] - offset


def _cache_graph_representation(
    representation: GraphRepresentation,
    dataset: DictDataset,
    *,
    jacobian_indices: Optional[torch.Tensor],
    batch_size: int,
    device: Optional[str | torch.device],
    output_device: str | torch.device,
    compute_jacobian: bool,
) -> RepresentationCache:
    if not representation.freeze:
        raise RuntimeError("Caching requires `representation.freeze=True`.")
    if dataset.metadata.get("data_type") != "graphs":
        raise TypeError("Expected a graph-based `DictDataset`.")
    if representation.output_kind != "system":
        raise ValueError(
            "Graph Jacobian caching currently requires a system-level representation."
        )
    if batch_size <= 0:
        raise ValueError("`batch_size` must be positive.")

    n_samples = len(dataset)
    selected = _resolve_indices(n_samples, jacobian_indices) if compute_jacobian else None
    original_device = _representation_device(representation)
    target_device = torch.device(device or original_device)
    output_device = torch.device(output_device)
    was_training = representation.training
    representation.to(target_device).eval()

    try:
        loader = GraphDataLoader(dataset, batch_size=batch_size, shuffle=False)
        feature_parts = []
        jacobian_parts = []
        expected_n_atoms = None
        offset = 0

        for batch in loader:
            graph = batch["data_list"].to(target_device)
            n_graphs = int(graph.num_graphs)
            local_selected = (
                _selected_local_indices(selected, offset, n_graphs)
                if compute_jacobian
                else torch.empty(0, dtype=torch.long)
            )

            if compute_jacobian and local_selected.numel() > 0:
                counts = torch.bincount(graph.batch, minlength=n_graphs)
                selected_counts = counts.index_select(
                    0, local_selected.to(device=counts.device)
                )
                if not torch.all(selected_counts == selected_counts[0]):
                    raise ValueError(
                        "Dense graph Jacobian caching requires equal atom counts "
                        "for selected graphs."
                    )
                n_atoms = int(selected_counts[0].item())
                if expected_n_atoms is None:
                    expected_n_atoms = n_atoms
                elif expected_n_atoms != n_atoms:
                    raise ValueError(
                        "All selected graphs must have the same atom count."
                    )

                graph.positions = graph.positions.detach().requires_grad_(True)
                latent = representation(graph).reshape(
                    n_graphs, representation.out_features
                )
                selected_device = local_selected.to(target_device)
                latent_selected = latent.index_select(0, selected_device)
                gradients = []

                for latent_index in range(representation.out_features):
                    gradient_positions = torch.autograd.grad(
                        latent_selected[:, latent_index].sum(),
                        graph.positions,
                        retain_graph=latent_index + 1 < representation.out_features,
                        create_graph=False,
                    )[0]
                    per_graph = []
                    for local_index in selected_device:
                        per_graph.append(gradient_positions[graph.batch == local_index])
                    gradients.append(torch.stack(per_graph, dim=0))

                jacobian_parts.append(
                    torch.stack(gradients, dim=-1).detach().to(output_device)
                )
            else:
                with torch.no_grad():
                    latent = representation(graph).reshape(
                        n_graphs, representation.out_features
                    )

            feature_parts.append(latent.detach().to(output_device))
            offset += n_graphs

        if not feature_parts:
            raise ValueError("The graph dataset is empty.")
        if offset != n_samples:
            raise RuntimeError("Graph cache sample counting is inconsistent.")

        features = torch.cat(feature_parts, dim=0)
        if not compute_jacobian:
            return RepresentationCache(features=features)

        if not jacobian_parts:
            raise RuntimeError("Failed to construct graph representation Jacobians.")

        assert selected is not None
        return RepresentationCache(
            features=features,
            jacobian=torch.cat(jacobian_parts, dim=0),
            reference_indices=_reference_map(n_samples, selected, output_device),
        )
    finally:
        representation.to(original_device)
        representation.train(was_training)


def precompute_representation_cache(
    representation: Representation,
    dataset: DictDataset,
    *,
    descriptor_derivatives: Optional[SmartDerivatives] = None,
    jacobian_indices: Optional[torch.Tensor] = None,
    source_ref_idx: Optional[torch.Tensor] = None,
    batch_size: Optional[int] = None,
    device: Optional[str | torch.device] = None,
    output_device: str | torch.device = "cpu",
    compute_jacobian: bool = True,
) -> RepresentationCache:
    """Cache any representation without embedding task-specific semantics."""
    if isinstance(representation, TensorRepresentation):
        return _cache_tensor_representation(
            representation,
            dataset,
            descriptor_derivatives=descriptor_derivatives,
            jacobian_indices=jacobian_indices,
            source_ref_idx=source_ref_idx,
            batch_size=1024 if batch_size is None else batch_size,
            device=device,
            output_device=output_device,
            compute_jacobian=compute_jacobian,
        )

    if isinstance(representation, GraphRepresentation):
        if descriptor_derivatives is not None:
            raise ValueError(
                "`descriptor_derivatives` must not be passed for graph representations."
            )
        if source_ref_idx is not None:
            raise ValueError("`source_ref_idx` applies only to tensor representations.")
        return _cache_graph_representation(
            representation,
            dataset,
            jacobian_indices=jacobian_indices,
            batch_size=256 if batch_size is None else batch_size,
            device=device,
            output_device=output_device,
            compute_jacobian=compute_jacobian,
        )

    raise TypeError("Unsupported representation type.")


def _variational_indices(
    labels: torch.Tensor,
    separate_boundary_dataset: bool,
) -> torch.Tensor:
    labels = labels.reshape(-1)
    if separate_boundary_dataset:
        indices = torch.nonzero(labels > 1, as_tuple=False).reshape(-1)
    else:
        indices = torch.arange(labels.numel(), dtype=torch.long)
    if indices.numel() == 0:
        raise ValueError("No variational samples were found.")
    return indices


def _extract_graph_scalar_field(dataset: DictDataset, field: str) -> torch.Tensor:
    values = []
    for i, graph in enumerate(dataset["data_list"]):
        if not hasattr(graph, field):
            raise KeyError(f"Graph {i} does not define `{field}`.")
        value = getattr(graph, field)
        value = value if torch.is_tensor(value) else torch.as_tensor(value)
        value = value.reshape(-1)
        if value.numel() != 1:
            raise ValueError(f"`{field}` must contain one scalar per graph.")
        values.append(value.detach().cpu())
    if not values:
        raise ValueError("The graph dataset is empty.")
    return torch.cat(values, dim=0)


def precompute_committor_cache(
    representation: Representation,
    dataset: DictDataset,
    descriptor_derivatives: Optional[SmartDerivatives] = None,
    batch_size: Optional[int] = None,
    device: Optional[str | torch.device] = None,
    output_device: str | torch.device = "cpu",
    separate_boundary_dataset: bool = True,
) -> tuple[DictDataset, CachedRepresentationDerivatives]:
    """Committor-specific adapter around :func:`precompute_representation_cache`."""
    output_device = torch.device(output_device)

    if isinstance(representation, TensorRepresentation):
        keys = _dataset_keys(dataset)
        required = ("data", "labels", "weights", "ref_idx")
        missing = [key for key in required if key not in keys]
        if missing:
            raise KeyError(f"Dataset is missing required keys: {missing}.")
        if descriptor_derivatives is None:
            raise ValueError(
                "`descriptor_derivatives` is required for tensor representations."
            )

        labels = dataset["labels"].reshape(-1)
        refs = dataset["ref_idx"].reshape(-1).long()
        indices = _variational_indices(labels, separate_boundary_dataset)
        cache = precompute_representation_cache(
            representation,
            dataset,
            descriptor_derivatives=descriptor_derivatives,
            jacobian_indices=indices,
            source_ref_idx=refs,
            batch_size=batch_size,
            device=device,
            output_device=output_device,
            compute_jacobian=True,
        )

        cached_data = {
            key: dataset[key].to(output_device) if torch.is_tensor(dataset[key]) else dataset[key]
            for key in keys
        }
        cached_data["data"] = cache.features
        cached_data["ref_idx"] = cache.reference_indices
        cached_dataset = DictDataset(cached_data)

    elif isinstance(representation, GraphRepresentation):
        labels = _extract_graph_scalar_field(dataset, "graph_labels")
        weights = _extract_graph_scalar_field(dataset, "weight")
        indices = _variational_indices(labels, separate_boundary_dataset)
        cache = precompute_representation_cache(
            representation,
            dataset,
            jacobian_indices=indices,
            batch_size=batch_size,
            device=device,
            output_device=output_device,
            compute_jacobian=True,
        )
        cached_dataset = DictDataset(
            {
                "data": cache.features,
                "labels": labels.to(output_device),
                "weights": weights.to(output_device),
                "ref_idx": cache.reference_indices,
            }
        )
    else:
        raise TypeError("Unsupported representation type.")

    if cache.jacobian is None or cache.reference_indices is None:
        raise RuntimeError("Representation caching did not produce derivatives.")

    return cached_dataset, CachedRepresentationDerivatives(cache.jacobian)