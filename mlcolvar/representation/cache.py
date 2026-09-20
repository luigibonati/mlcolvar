from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
from torch_geometric.loader import DataLoader as GraphDataLoader

from mlcolvar.core.loss.utils.smart_derivatives import SmartDerivatives
from mlcolvar.data import DictDataset

from .base import GraphRepresentation, Representation, VectorRepresentation


__all__ = [
    "RepresentationCache",
    "IdentityDescriptorDerivatives",
    "CachedRepresentationDerivatives",
    "precompute_representation_cache",
    "precompute_committor_cache",
]


@dataclass
class RepresentationCache:
    features: torch.Tensor
    jacobian: Optional[torch.Tensor] = None
    reference_indices: Optional[torch.Tensor] = None


class IdentityDescriptorDerivatives(SmartDerivatives):
    def __init__(self):
        nn.Module.__init__(self)

    def forward(self, gradient_descriptor, ref_idx=None):
        return gradient_descriptor.unsqueeze(1)


class CachedRepresentationDerivatives(SmartDerivatives):
    def __init__(self, jacobian: torch.Tensor):
        nn.Module.__init__(self)
        self.register_buffer("jacobian", jacobian, persistent=False)

    def forward(self, gradient_latent, ref_idx=None):
        if ref_idx is None:
            raise ValueError("`ref_idx` is required.")

        ref_idx = ref_idx.reshape(-1).to(self.jacobian.device, dtype=torch.long)
        if torch.any(ref_idx < 0) or torch.any(ref_idx >= len(self.jacobian)):
            raise IndexError("Invalid cached derivative index.")

        jacobian = self.jacobian[ref_idx]
        gradient_latent = gradient_latent.to(jacobian)

        return torch.einsum("bl,b...l->b...", gradient_latent, jacobian)


def _keys(dataset):
    keys = dataset.keys
    return tuple(keys() if callable(keys) else keys)


def _device(module):
    return next(
        iter(module.parameters()),
        next(iter(module.buffers()), torch.empty(0)),
    ).device


def _indices(n, indices=None):
    if indices is None:
        return torch.arange(n, dtype=torch.long)

    indices = torch.as_tensor(indices).cpu()

    if indices.dtype == torch.bool:
        indices = indices.reshape(-1)

        if len(indices) != n:
            raise ValueError(
                "Boolean `jacobian_indices` must have "
                "the same length as the dataset."
            )

        indices = torch.nonzero(
            indices,
            as_tuple=False,
        ).reshape(-1)

    else:
        indices = indices.reshape(-1).long()

    indices = torch.unique(
        indices,
        sorted=True,
    )

    if torch.any(indices < 0) or torch.any(indices >= n):
        raise IndexError(
            "`jacobian_indices` contains out-of-range indices."
        )

    return indices


def _references(n, selected, device):
    refs = torch.full((n,), -1, dtype=torch.long, device=device)
    selected = selected.to(device)
    refs[selected] = torch.arange(len(selected), device=device)
    return refs


def _prepare(representation, device):
    state = (_device(representation), representation.training)
    representation.to(device).eval()
    return state


def _restore(representation, state):
    device, training = state
    representation.to(device).train(training)


def _cache_vector(
    representation,
    dataset,
    *,
    descriptor_derivatives,
    jacobian_indices,
    source_ref_idx,
    batch_size,
    device,
    output_device,
    compute_jacobian,
):
    if not representation.freeze:
        raise RuntimeError("Caching requires a frozen representation.")

    keys = _keys(dataset)
    if "data" not in keys:
        raise KeyError("Vector caching requires `data`.")

    x = dataset["data"]
    cell = dataset["cell"] if "cell" in keys else None
    n = len(x)

    device = torch.device(device or _device(representation))
    output_device = torch.device(output_device)

    state = _prepare(representation, device)

    try:
        parts = []

        with torch.no_grad():
            for start in range(0, n, batch_size):
                stop = min(start + batch_size, n)
                xb = x[start:stop].to(device)
                cb = None if cell is None else cell[start:stop].to(device)

                h = representation(xb, cell=cb)
                
                parts.append(
                    h.reshape(len(xb), -1).detach().to(output_device)
                )

        features = torch.cat(parts)

        if not compute_jacobian:
            return RepresentationCache(features)

        if descriptor_derivatives is None:
            raise ValueError("`descriptor_derivatives` is required.")

        descriptor_derivatives.to(device)
        selected = _indices(n, jacobian_indices)

        if source_ref_idx is None:
            source_ref_idx = torch.arange(n)
        else:
            source_ref_idx = torch.as_tensor(source_ref_idx).reshape(-1).long().cpu()

        jacobians = []

        with torch.enable_grad():
            for start in range(0, len(selected), batch_size):
                idx = selected[start : start + batch_size]

                xb = (
                    x[idx.to(x.device)]
                    .to(device)
                    .detach()
                    .requires_grad_(True)
                )

                cb = (
                    None
                    if cell is None
                    else cell[idx.to(cell.device)].to(device)
                )

                refs = source_ref_idx[idx].to(device)

                h = representation(
                    xb,
                    cell=cb,
                ).reshape(len(xb), -1)

                grads = []

                for i in range(h.shape[-1]):
                    dh = torch.autograd.grad(
                        h[:, i].sum(),
                        xb,
                        retain_graph=i + 1 < h.shape[-1],
                    )[0]

                    grads.append(
                        descriptor_derivatives(dh, refs)
                    )

                jacobians.append(
                    torch.stack(grads, dim=-1).detach().to(output_device)
                )

        return RepresentationCache(
            features,
            torch.cat(jacobians),
            _references(n, selected, output_device),
        )

    finally:
        _restore(representation, state)


def _cache_graph(
    representation,
    dataset,
    *,
    jacobian_indices,
    batch_size,
    device,
    output_device,
    compute_jacobian,
):
    if not representation.freeze:
        raise RuntimeError("Caching requires a frozen representation.")

    if dataset.metadata.get("data_type") != "graphs":
        raise TypeError("Expected a graph dataset.")

    n = len(dataset)
    selected = _indices(n, jacobian_indices) if compute_jacobian else None

    device = torch.device(device or _device(representation))
    output_device = torch.device(output_device)

    state = _prepare(representation, device)

    try:
        loader = GraphDataLoader(dataset, batch_size=batch_size, shuffle=False)

        features = []
        jacobians = []
        offset = 0
        n_atoms_ref = None

        for batch in loader:
            graph = batch["data_list"].to(device)
            n_graphs = int(graph.num_graphs)

            local = (
                selected[(selected >= offset) & (selected < offset + n_graphs)]
                - offset
                if compute_jacobian
                else torch.empty(0, dtype=torch.long)
            )

            need_grad = compute_jacobian and len(local) > 0

            if need_grad:
                counts = torch.bincount(graph.batch, minlength=n_graphs)
                selected_counts = counts[local.to(counts.device)]

                if not torch.all(selected_counts == selected_counts[0]):
                    raise ValueError("Selected graphs must have equal atom counts.")

                n_atoms = int(selected_counts[0])
                if n_atoms_ref is None:
                    n_atoms_ref = n_atoms
                elif n_atoms != n_atoms_ref:
                    raise ValueError("Selected graphs must have equal atom counts.")

                graph.positions = graph.positions.detach().requires_grad_(True)

            with torch.enable_grad() if need_grad else torch.no_grad():
                h = representation(graph)

                if h.ndim != 2 or h.shape[0] != n_graphs:
                    raise ValueError(
                        "Cached features must contain one row per graph."
                    )

                if need_grad:
                    local = local.to(device)
                    hs = h[local]
                    grads = []

                    for i in range(h.shape[-1]):
                        dp = torch.autograd.grad(
                            hs[:, i].sum(),
                            graph.positions,
                            retain_graph=i + 1 < h.shape[-1],
                        )[0]

                        grads.append(
                            torch.stack(
                                [dp[graph.batch == j] for j in local]
                            )
                        )

                    jacobians.append(
                        torch.stack(grads, dim=-1)
                        .detach()
                        .to(output_device)
                    )

            features.append(h.detach().to(output_device))
            offset += n_graphs

        features = torch.cat(features)

        if not compute_jacobian:
            return RepresentationCache(features)

        return RepresentationCache(
            features,
            torch.cat(jacobians),
            _references(n, selected, output_device),
        )

    finally:
        _restore(representation, state)


def precompute_representation_cache(
    representation: Representation,
    dataset: DictDataset,
    *,
    descriptor_derivatives: Optional[SmartDerivatives] = None,
    jacobian_indices: Optional[torch.Tensor] = None,
    source_ref_idx: Optional[torch.Tensor] = None,
    batch_size: Optional[int] = None,
    device=None,
    output_device="cpu",
    compute_jacobian: bool = False,
):
    common = dict(
        jacobian_indices=jacobian_indices,
        device=device,
        output_device=output_device,
        compute_jacobian=compute_jacobian,
    )

    if isinstance(representation, VectorRepresentation):
        return _cache_vector(
            representation,
            dataset,
            descriptor_derivatives=descriptor_derivatives,
            source_ref_idx=source_ref_idx,
            batch_size=batch_size or 1024,
            **common,
        )

    if isinstance(representation, GraphRepresentation):
        if descriptor_derivatives is not None or source_ref_idx is not None:
            raise ValueError(
                "Descriptor derivatives and source indices are vector-only."
            )
            
        return _cache_graph(
            representation,
            dataset,
            batch_size=batch_size or 256,
            **common,
        )

    raise TypeError("Unsupported representation type.")


def _graph_field(dataset, field):
    values = []

    for graph in dataset["data_list"]:
        if not hasattr(graph, field):
            raise KeyError(f"Graph does not define `{field}`.")

        values.append(
            torch.as_tensor(getattr(graph, field)).reshape(-1).cpu()
        )

    return torch.cat(values)


def precompute_committor_cache(
    representation: Representation,
    dataset: DictDataset,
    descriptor_derivatives: Optional[SmartDerivatives] = None,
    batch_size=None,
    device=None,
    output_device="cpu",
    separate_boundary_dataset=True,
):
    output_device = torch.device(output_device)

    if isinstance(representation, VectorRepresentation):
        keys = _keys(dataset)
        required = ("data", "labels", "weights", "ref_idx")

        missing = [key for key in required if key not in keys]
        if missing:
            raise KeyError(f"Missing keys: {missing}")

        labels = dataset["labels"].reshape(-1)
        refs = dataset["ref_idx"].reshape(-1).long()

    elif isinstance(representation, GraphRepresentation):
        labels = _graph_field(dataset, "graph_labels")
        refs = None

    else:
        raise TypeError("Unsupported representation type.")

    indices = (
        torch.nonzero(labels > 1).reshape(-1)
        if separate_boundary_dataset
        else torch.arange(len(labels))
    )

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

    if isinstance(representation, VectorRepresentation):
        data = {
            key: value.to(output_device) if torch.is_tensor(value) else value
            for key, value in ((key, dataset[key]) for key in keys)
        }
        data["data"] = cache.features
        data["ref_idx"] = cache.reference_indices

    else:
        data = {
            "data": cache.features,
            "labels": labels.to(output_device),
            "weights": _graph_field(dataset, "weight").to(output_device),
            "ref_idx": cache.reference_indices,
        }

    return (
        DictDataset(data),
        CachedRepresentationDerivatives(cache.jacobian),
    )