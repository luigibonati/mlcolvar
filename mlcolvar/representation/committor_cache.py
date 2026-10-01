from typing import Optional

import torch

from mlcolvar.core.loss.utils.smart_derivatives import (
    SmartDerivatives,
    create_smart_dataset,
)
from mlcolvar.data import DictDataset

from .base import GraphRepresentation, Representation, VectorRepresentation
from .cache import CachedRepresentationDerivatives

__all__ = ["precompute_committor_cache"]


def _graph_field(dataset, field):
    return torch.cat([
        torch.as_tensor(getattr(graph, field)).reshape(-1)
        for graph in dataset["data_list"]
    ])


def precompute_committor_cache(
    representation: Representation,
    dataset: DictDataset,
    descriptor_derivatives: Optional[SmartDerivatives] = None,
    batch_size=None,
    device=None,
    output_device="cpu",
    separate_boundary_dataset=True,
):
    """Precompute representation features and Jacobians for committor training."""
    output_device = torch.device(output_device)

    if isinstance(representation, VectorRepresentation):
        required = {"data", "labels", "weights", "ref_idx"}
        missing = required.difference(dataset.keys)
        if missing:
            raise KeyError(f"Missing keys: {sorted(missing)}")

        labels = dataset["labels"].reshape(-1)
        source_ref_idx = dataset["ref_idx"].reshape(-1).long()

    elif isinstance(representation, GraphRepresentation):
        labels = _graph_field(dataset, "graph_labels")
        source_ref_idx = None

    else:
        raise TypeError("Unsupported representation type.")

    indices = (
        torch.nonzero(labels > 1).reshape(-1)
        if separate_boundary_dataset
        else torch.arange(len(labels))
    )

    cache = representation.cache(
        dataset,
        jacobian=True,
        descriptor_derivatives=descriptor_derivatives,
        jacobian_indices=indices,
        source_ref_idx=source_ref_idx,
        batch_size=batch_size,
        device=device,
        output_device=output_device,
    )

    if isinstance(representation, VectorRepresentation):
        cached_dataset = create_smart_dataset(
            cache.features,
            dataset,
            separate_boundary_dataset,
        )

    else:
        graph_dataset = DictDataset({
            "data": cache.features,
            "labels": labels.to(output_device),
            "weights": _graph_field(dataset, "weight").to(output_device),
        })

        cached_dataset = create_smart_dataset(
            cache.features,
            graph_dataset,
            separate_boundary_dataset,
        )

    return (
        cached_dataset,
        CachedRepresentationDerivatives(cache.jacobian),
    )