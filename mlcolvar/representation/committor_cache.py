from typing import Optional

import torch
from torch import nn

from mlcolvar.core.loss.utils.smart_derivatives import create_smart_dataset
from mlcolvar.data import DictDataset

from .base import Representation
from .derivatives import CachedRepresentationDerivatives

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
    descriptor_derivatives: Optional[nn.Module] = None,
    batch_size=None,
    device=None,
    output_device="cpu",
    separate_boundary_dataset=True,
):
    """Cache a frozen representation for committor training.

    The representation is evaluated once and its Jacobians are cached for
    samples contributing to the derivative-based committor loss.

    For vector representations, cached Jacobians are computed with respect
    to the representation inputs. If ``descriptor_derivatives`` is provided,
    they are subsequently transformed to Cartesian-coordinate Jacobians.

    For graph representations, Jacobians are computed directly with respect
    to atomic positions.

    Parameters
    ----------
    representation : Representation
        Frozen vector or graph representation.
    dataset : DictDataset
        Dataset containing the committor training data.
    descriptor_derivatives : torch.nn.Module, optional
        Transform gradients with respect to vector descriptors into gradients
        with respect to Cartesian coordinates. Only supported for vector
        representations.
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
    cached_derivatives : CachedRepresentationDerivatives
        Derivative transform backed by cached Cartesian Jacobians.
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

    elif representation.input_kind == "graph":
        if descriptor_derivatives is not None:
            raise ValueError(
                "`descriptor_derivatives` is only supported for "
                "vector representations."
            )

        labels = _graph_field(dataset, "graph_labels")

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
        jacobian_indices=indices,
        batch_size=batch_size,
        device=device,
        output_device=output_device,
    )

    jacobian = cache.jacobian

    if jacobian is None:
        raise RuntimeError(
            "Representation cache did not return Jacobians."
        )

    if descriptor_derivatives is not None:
        source_ref_idx = dataset["ref_idx"].reshape(-1).long()
        selected_ref_idx = source_ref_idx[
            indices.to(source_ref_idx.device)
        ].to(jacobian.device)

        jacobian = descriptor_derivatives(
            jacobian,
            selected_ref_idx,
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
            "weights": _graph_field(
                dataset,
                "weight",
            ).to(output_device),
        })

        cached_dataset = create_smart_dataset(
            cache.features,
            graph_dataset,
            separate_boundary_dataset,
        )

    return (
        cached_dataset,
        CachedRepresentationDerivatives(jacobian),
    )