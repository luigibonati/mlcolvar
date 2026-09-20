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