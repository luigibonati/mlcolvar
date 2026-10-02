from typing import Optional

import torch
from torch import nn

from mlcolvar.core.loss.utils.smart_derivatives import (
    SmartDerivatives,
    create_smart_dataset,
)
from mlcolvar.data import DictDataset

from .derivatives import JacobianTransform, compute_jacobian

__all__ = ["prepare_committor"]


class _CommittorJacobianTransform(SmartDerivatives):
    """Adapt a precomputed representation Jacobian to the committor interface."""

    def __init__(self, jacobian: torch.Tensor):
        super().__init__()
        self.transform = JacobianTransform(jacobian)

    @property
    def jacobian(self) -> torch.Tensor:
        """Return the stored representation Jacobian."""
        return self.transform.jacobian

    def forward(self, gradient, ref_idx=None):
        """Propagate feature-space gradients through the stored Jacobian."""
        return self.transform(gradient, ref_idx)


def _graph_field(dataset: DictDataset, field: str) -> torch.Tensor:
    """Concatenate a per-graph field into a single tensor."""
    return torch.cat(
        [
            torch.as_tensor(graph[field]).reshape(-1)
            for graph in dataset["data_list"]
        ]
    )


def prepare_committor(
    model: nn.Module,
    dataset: DictDataset,
    features: torch.Tensor,
    descriptor_derivatives: Optional[nn.Module] = None,
    batch_size: Optional[int] = None,
    device=None,
    output_device="cpu",
    separate_boundary_dataset: bool = True,
):
    """Prepare precomputed representations and derivatives for committor training.

    The representation values are supplied through ``features`` while the
    corresponding Jacobians with respect to the original model inputs are
    computed with :func:`compute_jacobian`. The resulting derivative transform
    allows committor gradients evaluated in representation space to be
    propagated back to the original coordinates.

    Parameters
    ----------
    model : torch.nn.Module
        Representation model used to compute the Jacobian.
    dataset : DictDataset
        Original vector- or graph-based dataset.
    features : torch.Tensor
        Precomputed representation values for all dataset samples.
    descriptor_derivatives : torch.nn.Module, optional
        Optional transform used to propagate representation Jacobians through
        descriptor derivatives for vector datasets. Not supported for graph
        datasets.
    batch_size : int, optional
        Batch size used to evaluate representation Jacobians.
    device : torch.device or str, optional
        Device used for Jacobian evaluation.
    output_device : torch.device or str, default="cpu"
        Device on which the computed Jacobian is stored.
    separate_boundary_dataset : bool, default=True
        Whether boundary samples are separated from transition-region samples
        following the committor smart-dataset convention.

    Returns
    -------
    tuple
        Prepared committor dataset and a derivative transform containing the
        corresponding precomputed representation Jacobian.
    """
    is_graph = dataset.metadata.get("data_type") == "graphs"

    if is_graph:
        if descriptor_derivatives is not None:
            raise ValueError(
                "`descriptor_derivatives` is not supported for graph datasets."
            )

        source_dataset = DictDataset(
            {
                "data": features,
                "labels": _graph_field(dataset, "graph_labels").to(features.device),
                "weights": _graph_field(dataset, "weight").to(features.device),
            }
        )
    else:
        required = {"data", "labels", "weights"}
        if descriptor_derivatives is not None:
            required.add("ref_idx")

        missing = required.difference(dataset.keys)
        if missing:
            raise KeyError(f"Missing keys: {sorted(missing)}")

        source_dataset = dataset

    labels = source_dataset["labels"].reshape(-1)
    if len(features) != len(labels):
        raise ValueError(
            "The number of features must match the number of dataset samples."
        )

    # Jacobians are required only for the samples used by the derivative loss.
    indices = (
        torch.nonzero(labels > 1, as_tuple=False).reshape(-1)
        if separate_boundary_dataset
        else torch.arange(len(labels))
    )

    jacobian = compute_jacobian(
        model,
        dataset,
        indices=indices,
        batch_size=batch_size,
        device=device,
        output_device=output_device,
    )

    # For descriptor-based inputs, apply the chain rule through the descriptor
    # derivatives to obtain derivatives with respect to the original coordinates.
    if descriptor_derivatives is not None:
        ref_idx = dataset["ref_idx"].reshape(-1).long()
        ref_idx = ref_idx[indices.to(ref_idx.device)].to(jacobian.device)
        jacobian = descriptor_derivatives.to(jacobian.device)(
            jacobian,
            ref_idx,
        )

    prepared_dataset = create_smart_dataset(
        features,
        source_dataset,
        separate_boundary_dataset,
    )

    return prepared_dataset, _CommittorJacobianTransform(jacobian)