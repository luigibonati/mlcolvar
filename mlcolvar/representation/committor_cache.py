from typing import Optional

import torch

from mlcolvar.core.loss.utils.smart_derivatives import (
    SmartDerivatives,
    create_smart_dataset,
)
from mlcolvar.data import DictDataset

from .base import Representation
from .cache import CachedRepresentationDerivatives

__all__ = ["precompute_committor_cache"]


def _graph_field(dataset, field):
    """Concatenate a graph-level field across all systems."""
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
    """Cache a frozen representation for committor training.

    The representation is evaluated once and its Jacobians are cached for
    samples contributing to the derivative-based committor loss.

    Parameters
    ----------
    representation : Representation
        Frozen vector or graph representation.
    dataset : DictDataset
        Dataset containing the committor training data.
    descriptor_derivatives : SmartDerivatives, optional
        Transform descriptor gradients to Cartesian-coordinate gradients.
        If None, vector inputs are treated directly as coordinates.
    batch_size : int, optional
        Batch size used to evaluate the representation.
    device : str or torch.device, optional
        Device used for representation evaluation.
    output_device : str or torch.device, default="cpu"
        Device used to store cached tensors.
    separate_boundary_dataset : bool, default=True
        If True, cache Jacobians only for samples with ``labels > 1``.

    Returns
    -------
    cached_dataset : DictDataset
        Dataset containing cached representation features and committor data.
    descriptor_derivatives : CachedRepresentationDerivatives
        Derivative transform backed by cached representation Jacobians.
    """
    output_device = torch.device(output_device)

    if representation.input_kind == "vector":
        required = {"data", "labels", "weights"}
        if descriptor_derivatives is not None:
            required.add("ref_idx")

        missing = required.difference(dataset.keys)
        if missing:
            raise KeyError(f"Missing keys: {sorted(missing)}")

        labels = dataset["labels"].reshape(-1)
        source_ref_idx = (
            dataset["ref_idx"].reshape(-1).long()
            if descriptor_derivatives is not None
            else None
        )

    elif representation.input_kind == "graph":
        labels = _graph_field(dataset, "graph_labels")
        source_ref_idx = None

    else:
        raise ValueError(
            "Unsupported representation input kind: "
            f"{representation.input_kind!r}."
        )

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

    if representation.input_kind == "vector":
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