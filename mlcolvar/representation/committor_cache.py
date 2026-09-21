from typing import Optional

import torch

from mlcolvar.core.loss.utils.smart_derivatives import SmartDerivatives
from mlcolvar.data import DictDataset

from .base import GraphRepresentation, Representation, VectorRepresentation
from .cache import CachedRepresentationDerivatives


__all__ = [
    "precompute_committor_cache",
]


def _keys(dataset):
    keys = dataset.keys
    return tuple(keys() if callable(keys) else keys)


def _graph_field(dataset, field):
    values = []

    for graph in dataset["data_list"]:
        if not hasattr(graph, field):
            raise KeyError(
                f"Graph does not define `{field}`."
            )

        values.append(
            torch.as_tensor(
                getattr(graph, field)
            )
            .reshape(-1)
            .cpu()
        )

    return torch.cat(values)


def precompute_committor_cache(
    representation: Representation,
    dataset: DictDataset,
    descriptor_derivatives: Optional[
        SmartDerivatives
    ] = None,
    batch_size=None,
    device=None,
    output_device="cpu",
    separate_boundary_dataset=True,
):
    """Precompute representation features and Jacobians for committor training.

    The representation must be frozen. For vector representations,
    ``descriptor_derivatives`` maps descriptor gradients to coordinate
    derivatives. Graph representations compute coordinate Jacobians
    directly from atomic positions.

    Parameters
    ----------
    representation
        Frozen vector or graph representation.
    dataset
        Dataset used for committor training.
    descriptor_derivatives
        Descriptor derivative model required for vector representations.
    batch_size
        Batch size used during caching.
    device
        Device used to evaluate the representation.
    output_device
        Device where cached tensors are stored.
    separate_boundary_dataset
        If True, cache Jacobians only for samples with labels greater
        than one.

    Returns
    -------
    cached_dataset
        Dataset containing cached representation features.
    derivatives
        Derivative model backed by the cached representation Jacobians.

    Notes
    -----
    Graph Jacobian caching currently requires all selected graphs to
    contain the same number of atoms.
    """
    output_device = torch.device(
        output_device
    )

    if isinstance(
        representation,
        VectorRepresentation,
    ):
        keys = _keys(dataset)
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
                f"Missing keys: {missing}"
            )

        labels = dataset[
            "labels"
        ].reshape(-1)

        refs = dataset[
            "ref_idx"
        ].reshape(-1).long()

    elif isinstance(
        representation,
        GraphRepresentation,
    ):
        labels = _graph_field(
            dataset,
            "graph_labels",
        )
        refs = None

    else:
        raise TypeError(
            "Unsupported representation type."
        )

    indices = (
        torch.nonzero(
            labels > 1
        ).reshape(-1)
        if separate_boundary_dataset
        else torch.arange(len(labels))
    )

    cache = representation.cache(
        dataset,
        jacobian=True,
        descriptor_derivatives=(
            descriptor_derivatives
        ),
        jacobian_indices=indices,
        source_ref_idx=refs,
        batch_size=batch_size,
        device=device,
        output_device=output_device,
    )

    if isinstance(
        representation,
        VectorRepresentation,
    ):
        data = {
            key: (
                value.to(output_device)
                if torch.is_tensor(value)
                else value
            )
            for key, value in (
                (key, dataset[key])
                for key in keys
            )
        }

        data["data"] = cache.features
        data[
            "ref_idx"
        ] = cache.reference_indices

    else:
        data = {
            "data": cache.features,
            "labels": labels.to(
                output_device
            ),
            "weights": _graph_field(
                dataset,
                "weight",
            ).to(output_device),
            "ref_idx": (
                cache.reference_indices
            ),
        }

    return (
        DictDataset(data),
        CachedRepresentationDerivatives(
            cache.jacobian
        ),
    )