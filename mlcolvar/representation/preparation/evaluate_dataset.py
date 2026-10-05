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
    """Evaluate a model over a dataset and collect its outputs."""
    reference = module_reference_tensor(model)
    original_device = reference.device
    training = model.training
    device = original_device if device is None else torch.device(device)
    output_device = torch.device(output_device)

    is_graph = dataset.metadata.get("data_type") == "graphs"
    if batch_size is None:
        batch_size = 256 if is_graph else 1024
    loader = (
        GraphDataLoader(dataset, batch_size=batch_size, shuffle=False)
        if is_graph
        else DictLoader(dataset, batch_size=batch_size, shuffle=False)
    )

    outputs: list[torch.Tensor] = []
    model.to(device).eval()
    try:
        with torch.no_grad():
            for batch in loader:
                if is_graph:
                    data = batch["data_list"].to(device)
                    n_samples = int(data.num_graphs)
                    output = model(data)
                else:
                    data = batch["data"].to(device)
                    cell = batch.get("cell")
                    cell = None if cell is None else cell.to(device)
                    n_samples = len(data)
                    output = (
                        model(data)
                        if cell is None
                        else model(data, cell=cell)
                    )

                if not torch.is_tensor(output):
                    raise TypeError("`model` must return a torch.Tensor.")
                if output.ndim == 0 or output.shape[0] != n_samples:
                    raise ValueError(
                        "Model outputs must contain one row per sample."
                    )
                outputs.append(
                    output.reshape(n_samples, -1).detach().to(output_device)
                )
    finally:
        model.to(original_device).train(training)

    return torch.cat(outputs)