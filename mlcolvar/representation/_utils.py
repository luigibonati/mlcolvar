from typing import Any, Dict, Sequence

import torch
from torch import nn


__all__ = [
    "as_positive_int",
    "module_reference_tensor",
    "infer_num_graphs",
    "align_node_attrs",
]


def as_positive_int(value: Any, name: str) -> int:
    """Convert a scalar value to a positive Python integer."""
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError(f"`{name}` must be scalar.")
        value = value.detach().cpu().item()

    value = int(value)

    if value <= 0:
        raise ValueError(
            f"`{name}` must be positive. Found {value}."
        )

    return value


def module_reference_tensor(
    module: nn.Module,
) -> torch.Tensor:
    """Return a scalar tensor matching a module floating dtype/device."""
    for tensor in module.parameters():
        if tensor.is_floating_point() or tensor.is_complex():
            return torch.empty(
                (),
                dtype=tensor.dtype,
                device=tensor.device,
            )

    for tensor in module.buffers():
        if tensor.is_floating_point() or tensor.is_complex():
            return torch.empty(
                (),
                dtype=tensor.dtype,
                device=tensor.device,
            )

    return torch.empty(())


def infer_num_graphs(
    data: Dict[str, torch.Tensor],
) -> int:
    """Infer the number of systems represented by a graph dictionary."""
    if "ptr" in data:
        return data["ptr"].numel() - 1

    if "n_system" in data:
        value = data["n_system"]
        return (
            int(value.item())
            if value.ndim == 0
            else value.numel()
        )

    if "batch" in data:
        batch = data["batch"]
        return (
            0
            if batch.numel() == 0
            else int(batch.max().item()) + 1
        )

    raise RuntimeError(
        "Graph data must contain `ptr`, "
        "`n_system`, or `batch`."
    )


def _as_atomic_number_list(
    atomic_numbers: Sequence[int] | torch.Tensor,
    name: str,
) -> list[int]:
    values = (
        torch.as_tensor(
            atomic_numbers,
            dtype=torch.long,
        )
        .detach()
        .cpu()
        .reshape(-1)
        .tolist()
    )

    if not values:
        raise ValueError(
            f"The {name} atomic-number table cannot be empty."
        )

    if any(number <= 0 for number in values):
        raise ValueError(
            f"The {name} atomic-number table "
            "must contain positive integers."
        )

    if len(set(values)) != len(values):
        raise ValueError(
            f"The {name} atomic-number table "
            f"contains duplicates: {values}."
        )

    return values


def align_node_attrs(
    dataset,
    target_atomic_numbers: Sequence[int] | torch.Tensor,
):
    """Align graph one-hot node attributes with a representation element table."""
    if not hasattr(dataset, "metadata"):
        raise TypeError(
            "The dataset must expose a `metadata` attribute."
        )

    if "atomic_numbers" not in dataset.metadata:
        raise KeyError(
            "The dataset metadata must contain `atomic_numbers`."
        )

    source_values = _as_atomic_number_list(
        dataset.metadata["atomic_numbers"],
        "source",
    )
    target_values = _as_atomic_number_list(
        target_atomic_numbers,
        "target",
    )

    if source_values == target_values:
        return dataset

    target_indices = {
        number: index
        for index, number in enumerate(target_values)
    }

    missing = [
        number
        for number in source_values
        if number not in target_indices
    ]

    if missing:
        raise ValueError(
            "The target representation does not support "
            f"atomic numbers {missing}."
        )

    source_to_target = torch.tensor(
        [
            target_indices[number]
            for number in source_values
        ],
        dtype=torch.long,
    )

    for graph in dataset["data_list"]:
        node_attrs = graph["node_attrs"]

        if node_attrs.size(1) != len(source_values):
            raise ValueError(
                "`node_attrs` width does not match "
                "`dataset.metadata['atomic_numbers']`."
            )

        species = source_to_target.to(
            node_attrs.device
        )[node_attrs.argmax(dim=-1)]

        aligned = node_attrs.new_zeros(
            node_attrs.size(0),
            len(target_values),
        )
        aligned.scatter_(
            1,
            species.unsqueeze(1),
            1,
        )

        graph["node_attrs"] = aligned

    dataset.metadata[
        "atomic_numbers"
    ] = target_values

    return dataset