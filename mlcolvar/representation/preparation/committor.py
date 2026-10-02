from typing import Optional

import torch
from torch import nn

from mlcolvar.core.loss.utils.smart_derivatives import (
    SmartDerivatives,
    create_smart_dataset,
)
from mlcolvar.data import DictDataset

from ..base import Representation
from .derivatives import JacobianTransform, compute_jacobian

__all__ = ["prepare_committor"]


class _CommittorJacobianTransform(SmartDerivatives):
    """Adapt JacobianTransform to the committor derivative interface."""

    def __init__(self, jacobian: torch.Tensor):
        super().__init__()
        self.transform = JacobianTransform(jacobian)

    @property
    def jacobian(self) -> torch.Tensor:
        return self.transform.jacobian

    def forward(self, gradient, ref_idx=None):
        return self.transform(gradient, ref_idx)


def _graph_field(
    dataset: DictDataset,
    field: str,
) -> torch.Tensor:
    return torch.cat([
        torch.as_tensor(getattr(graph, field)).reshape(-1)
        for graph in dataset["data_list"]
    ])


def prepare_committor(
    representation: Representation,
    dataset: DictDataset,
    features: torch.Tensor,
    descriptor_derivatives: Optional[nn.Module] = None,
    batch_size: Optional[int] = None,
    device=None,
    output_device="cpu",
    separate_boundary_dataset: bool = True,
):
    """Prepare materialized representation features for committor training."""
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
                "`descriptor_derivatives` is only supported "
                "for vector representations."
            )

        labels = _graph_field(dataset, "graph_labels")

    else:
        raise ValueError(
            "Unsupported representation input kind: "
            f"{representation.input_kind!r}."
        )

    if len(features) != len(labels):
        raise ValueError(
            "The number of materialized features must match "
            "the number of dataset samples."
        )

    indices = (
        torch.nonzero(labels > 1, as_tuple=False).reshape(-1)
        if separate_boundary_dataset
        else torch.arange(len(labels))
    )

    jacobian = compute_jacobian(
        representation,
        dataset,
        indices=indices,
        batch_size=batch_size,
        device=device,
        output_device=output_device,
    )

    if descriptor_derivatives is not None:
        source_ref_idx = dataset["ref_idx"].reshape(-1).long()
        selected_ref_idx = source_ref_idx[
            indices.to(source_ref_idx.device)
        ].to(jacobian.device)

        descriptor_derivatives = descriptor_derivatives.to(jacobian.device)
        jacobian = descriptor_derivatives(
            jacobian,
            selected_ref_idx,
        )

    if representation.input_kind == "vector":
        feature_dataset = create_smart_dataset(
            features,
            dataset,
            separate_boundary_dataset,
        )
    else:
        graph_dataset = DictDataset({
            "data": features,
            "labels": labels.to(output_device),
            "weights": _graph_field(dataset, "weight").to(output_device),
        })

        feature_dataset = create_smart_dataset(
            features,
            graph_dataset,
            separate_boundary_dataset,
        )

    return feature_dataset, _CommittorJacobianTransform(jacobian)