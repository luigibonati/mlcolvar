# TODO: Move this module to `mlcolvar.cvs.committor.preparation`, as it contains committor-specific preparation logic rather than generic representation utilities?
import torch
from torch import nn
from torch_geometric.loader import DataLoader as GraphDataLoader

from mlcolvar.core.loss.utils.smart_derivatives import create_smart_dataset
from mlcolvar.data import DictDataset, DictLoader

from .._utils import module_reference_tensor

__all__ = ["prepare_committor"]


def _graph_field(dataset: DictDataset, field: str) -> torch.Tensor:
    return torch.cat([
        torch.as_tensor(graph[field]).reshape(-1)
        for graph in dataset["data_list"]
    ])


def _precompute_jacobian(
    model: nn.Module,
    dataset: DictDataset,
    indices: torch.Tensor,
    batch_size: int | None,
    device: torch.device | str | None,
    output_device: torch.device | str,
) -> torch.Tensor:
    """Precompute representation Jacobians in batches."""
    if indices.numel() == 0:
        raise ValueError("No samples selected for Jacobian computation.")

    dataset = dataset[indices.cpu()]
    is_graph = dataset.metadata.get("data_type") == "graphs"

    if batch_size is None:
        batch_size = 256 if is_graph else 1024
    if batch_size <= 0:
        raise ValueError("`batch_size` must be positive.")

    loader_cls = GraphDataLoader if is_graph else DictLoader
    loader = loader_cls(dataset, batch_size=batch_size, shuffle=False)

    original_device = module_reference_tensor(model).device
    training = model.training
    device = original_device if device is None else torch.device(device)

    jacobians = []
    n_atoms_ref = None

    try:
        model.to(device).eval()

        with torch.enable_grad():
            for batch in loader:
                if is_graph:
                    x = batch["data_list"].to(device)
                    counts = x.ptr[1:] - x.ptr[:-1]

                    if not torch.all(counts == counts[0]):
                        raise ValueError("Graphs must have equal atom counts.")

                    n_atoms = int(counts[0])
                    if n_atoms_ref is not None and n_atoms != n_atoms_ref:
                        raise ValueError("Graphs must have equal atom counts.")
                    n_atoms_ref = n_atoms

                    x.positions = x.positions.detach().requires_grad_(True)
                    inputs = x.positions
                    n_samples = x.num_graphs
                    output = model(x)
                else:
                    x = batch["data"].to(device).detach().requires_grad_(True)
                    cell = batch.get("cell")
                    inputs = x
                    n_samples = len(x)
                    output = (
                        model(x, cell=cell.to(device))
                        if cell is not None
                        else model(x)
                    )

                if (
                    not isinstance(output, torch.Tensor)
                    or output.ndim == 0
                    or output.shape[0] != n_samples
                ):
                    raise ValueError("Expected one output per sample.")

                output = output.reshape(n_samples, -1)

                gradients = [
                    torch.autograd.grad(
                        output[:, i].sum(),
                        inputs,
                        retain_graph=i < output.shape[1] - 1,
                    )[0].reshape(n_samples, -1)
                    for i in range(output.shape[1])
                ]

                jacobians.append(
                    torch.stack(gradients, dim=-1)
                    .detach()
                    .to(output_device)
                )
    finally:
        model.to(original_device).train(training)

    return torch.cat(jacobians)


def prepare_committor(
    model: nn.Module,
    dataset: DictDataset,
    features: torch.Tensor,
    descriptor_derivatives: nn.Module | None = None,
    batch_size: int | None = None,
    device: torch.device | str | None = None,
    output_device: torch.device | str = "cpu",
    separate_boundary_dataset: bool = True,
) -> tuple[DictDataset, torch.Tensor]:
    """Prepare precomputed features and Jacobians for committor training."""
    is_graph = dataset.metadata.get("data_type") == "graphs"

    if is_graph:
        if descriptor_derivatives is not None:
            raise ValueError(
                "`descriptor_derivatives` is not supported for graphs."
            )
        source_dataset = DictDataset({
            "data": features,
            "labels": _graph_field(dataset, "graph_labels").to(features.device),
            "weights": _graph_field(dataset, "weight").to(features.device),
        })
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
        raise ValueError("Features and dataset must have the same length.")

    indices = (
        torch.where(labels > 1)[0]
        if separate_boundary_dataset
        else torch.arange(len(labels))
    )

    jacobian = _precompute_jacobian(
        model, dataset, indices, batch_size, device, output_device
    )

    if descriptor_derivatives is not None:
        ref_idx = dataset["ref_idx"].reshape(-1)
        ref_idx = ref_idx[indices.to(ref_idx.device)]
        jacobian = descriptor_derivatives.to(jacobian.device)(
            jacobian, ref_idx.to(jacobian.device)
        )
        jacobian = jacobian.reshape(len(indices), -1, features.shape[-1])

    prepared_dataset = create_smart_dataset(
        features,
        source_dataset,
        separate_boundary_dataset,
    )

    return prepared_dataset, jacobian
