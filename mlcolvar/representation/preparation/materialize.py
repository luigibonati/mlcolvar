from typing import Optional

import torch
from torch_geometric.loader import DataLoader as GraphDataLoader

from mlcolvar.data import DictDataset

from ..base import Representation
from ._utils import require_graph_dataset, resolve_devices, temporary_eval

__all__ = ["materialize"]


def _materialize_vector(
    representation: Representation,
    dataset: DictDataset,
    batch_size: int,
    device: torch.device,
    output_device: torch.device,
) -> torch.Tensor:
    if "data" not in dataset.keys:
        raise KeyError("Vector materialization requires `data`.")

    x = dataset["data"]
    cell = dataset["cell"] if "cell" in dataset.keys else None
    outputs = []

    with temporary_eval(representation, device), torch.no_grad():
        for start in range(0, len(x), batch_size):
            xb = x[start : start + batch_size].to(device)
            cb = (
                None
                if cell is None
                else cell[start : start + batch_size].to(device)
            )

            output = representation(xb, cell=cb)
            outputs.append(
                output.reshape(len(xb), -1).detach().to(output_device)
            )

    return torch.cat(outputs)


def _materialize_graph(
    representation: Representation,
    dataset: DictDataset,
    batch_size: int,
    device: torch.device,
    output_device: torch.device,
) -> torch.Tensor:
    require_graph_dataset(dataset)

    loader = GraphDataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
    )
    outputs = []

    with temporary_eval(representation, device), torch.no_grad():
        for batch in loader:
            graph = batch["data_list"].to(device)
            output = representation(graph)

            if output.ndim != 2 or output.shape[0] != graph.num_graphs:
                raise ValueError(
                    "Materialized outputs must contain one row per graph."
                )

            outputs.append(output.detach().to(output_device))

    return torch.cat(outputs)


def materialize(
    representation: Representation,
    dataset: DictDataset,
    *,
    batch_size: Optional[int] = None,
    device=None,
    output_device="cpu",
) -> torch.Tensor:
    """Materialize a frozen representation over a dataset."""
    if not isinstance(representation, Representation):
        raise TypeError(
            "`representation` must derive from `Representation`."
        )

    if not representation.freeze:
        raise RuntimeError(
            "Materialization requires a frozen representation."
        )

    device, output_device = resolve_devices(
        representation,
        device,
        output_device,
    )

    if representation.input_kind == "vector":
        return _materialize_vector(
            representation,
            dataset,
            batch_size or 1024,
            device,
            output_device,
        )

    if representation.input_kind == "graph":
        if representation.output_kind != "system":
            raise ValueError(
                "Graph materialization requires system-level "
                "representation outputs. Use pooling or "
                "`concat_atoms()` first."
            )

        return _materialize_graph(
            representation,
            dataset,
            batch_size or 256,
            device,
            output_device,
        )

    raise ValueError(
        "Unsupported representation input kind: "
        f"{representation.input_kind!r}."
    )