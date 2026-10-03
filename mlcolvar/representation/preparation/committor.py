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
    """Adapt a representation Jacobian to the committor interface."""

    def __init__(self, jacobian: torch.Tensor):
        super().__init__()
        self.transform = JacobianTransform(jacobian)

    @property
    def jacobian(self) -> torch.Tensor:
        return self.transform.jacobian

    def forward(
        self,
        gradient: torch.Tensor,
        ref_idx: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.transform(gradient, ref_idx)


def _graph_field(dataset: DictDataset, field: str) -> torch.Tensor:
    """Concatenate a per-graph field."""
    return torch.cat(
        [torch.as_tensor(graph[field]).reshape(-1) for graph in dataset["data_list"]]
    )


def prepare_committor(
    model: nn.Module,
    dataset: DictDataset,
    features: torch.Tensor,
    descriptor_derivatives: nn.Module | None = None,
    batch_size: int | None = None,
    device: torch.device | str | None = None,
    output_device: torch.device | str = "cpu",
    separate_boundary_dataset: bool = True,
) -> tuple[DictDataset, _CommittorJacobianTransform]:
    """Prepare precomputed representations and Jacobians for committor training."""
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

    indices = (
        torch.where(labels > 1)[0]
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

    if descriptor_derivatives is not None:
        ref_idx = dataset["ref_idx"].reshape(-1).long()
        ref_idx = ref_idx[indices.to(ref_idx.device)].to(jacobian.device)
        jacobian = descriptor_derivatives.to(jacobian.device)(jacobian, ref_idx)

    return (
        create_smart_dataset(
            features,
            source_dataset,
            separate_boundary_dataset,
        ),
        _CommittorJacobianTransform(jacobian),
    )