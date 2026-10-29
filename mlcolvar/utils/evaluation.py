from itertools import chain

import torch
from torch import nn
from torch_geometric.loader import DataLoader as GraphDataLoader

from mlcolvar.data import DictDataset, DictLoader

__all__ = ["evaluate_dataset"]


def module_reference_tensor(module: nn.Module) -> torch.Tensor:
    """Find a reference tensor for the model's dtype and device."""
    fallback = None
    for tensor in chain(module.parameters(), module.buffers()):
        if fallback is None:
            fallback = tensor
        if tensor.is_floating_point() or tensor.is_complex():
            return tensor
    if fallback is not None:
        return fallback.new_empty((), dtype=torch.get_default_dtype())
    return torch.empty(())


def evaluate_dataset(
    model: nn.Module,
    dataset: DictDataset,
    *,
    batch_size: int | None = None,
    device: torch.device | str | None = None,
    output_device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Evaluate a model on a dataset in batches."""
    if not len(dataset):
        raise ValueError("Dataset must not be empty.")

    is_graph = dataset.metadata.get("data_type") == "graphs"
    batch_size = (256 if is_graph else 1024) if batch_size is None else batch_size
    if batch_size <= 0:
        raise ValueError("`batch_size` must be positive.")

    loader = (GraphDataLoader if is_graph else DictLoader)(
        dataset, batch_size=batch_size, shuffle=False
    )
    original_device = module_reference_tensor(model).device
    device = original_device if device is None else torch.device(device)
    training = model.training
    outputs = []

    try:
        model.to(device).eval()
        with torch.no_grad():
            for batch in loader:
                x = batch["data_list" if is_graph else "data"].to(device)
                cell = None if is_graph else batch.get("cell")
                y = model(x, cell=cell.to(device)) if cell is not None else model(x)
                n = x.num_graphs if is_graph else len(x)

                if not isinstance(y, torch.Tensor) or y.ndim == 0 or y.shape[0] != n:
                    raise ValueError("Expected one output per sample.")
                outputs.append(y.reshape(n, -1).to(output_device))
    finally:
        model.to(original_device).train(training)

    return torch.cat(outputs)
