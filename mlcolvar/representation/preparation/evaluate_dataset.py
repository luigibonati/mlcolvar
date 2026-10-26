# TODO: Move this module to `mlcolvar.utils.evaluation`, as dataset evaluation is not specific to representations?
import torch
from torch import nn
from torch_geometric.loader import DataLoader as GraphDataLoader

from mlcolvar.data import DictDataset, DictLoader
from .._utils import module_reference_tensor

__all__ = ["evaluate_dataset"]


def evaluate_dataset(
    model: nn.Module,
    dataset: DictDataset,
    *,
    batch_size: int | None = None,
    device: torch.device | str | None = None,
    output_device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Evaluate a model on a dataset in batches."""
    original_device = module_reference_tensor(model).device
    training = model.training
    device = original_device if device is None else torch.device(device)

    is_graph = dataset.metadata.get("data_type") == "graphs"
    loader_cls = GraphDataLoader if is_graph else DictLoader
    if batch_size is None:
        batch_size = 256 if is_graph else 1024
    loader = loader_cls(dataset, batch_size=batch_size, shuffle=False)

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
                    raise ValueError("Model must return one tensor row per sample.")

                outputs.append(y.reshape(n, -1).to(output_device))
    finally:
        model.to(original_device).train(training)

    return torch.cat(outputs)
