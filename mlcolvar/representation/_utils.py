
from collections.abc import Sequence
from typing import Any

import torch

__all__ = [
    "as_int",
    "as_float",
    "as_positive_int",
    "module_reference_tensor",
    "align_node_attrs",
]


def _scalar(value: Any, name: str):
    """Convert a scalar tensor to a Python scalar."""
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError(f"`{name}` must be scalar.")
        return value.item()
    return value


def as_int(value: Any, name: str) -> int:
    """Convert a scalar value to an integer."""
    return int(_scalar(value, name))


def as_float(value: Any, name: str) -> float:
    """Convert a scalar value to a float."""
    return float(_scalar(value, name))


def as_positive_int(value: Any, name: str) -> int:
    """Convert a scalar value to a positive integer."""
    value = as_int(value, name)
    if value <= 0:
        raise ValueError(f"`{name}` must be positive. Found {value}.")
    return value


def _as_atomic_number_list(
    atomic_numbers: Sequence[int] | torch.Tensor,
    name: str,
) -> list[int]:
    """Convert and validate atomic numbers."""
    values = torch.as_tensor(
        atomic_numbers, dtype=torch.long
    ).flatten().tolist()

    if not values or any(z <= 0 for z in values):
        raise ValueError(f"The {name} atomic numbers must be positive and non-empty.")
    if len(values) != len(set(values)):
        raise ValueError(f"The {name} atomic numbers contain duplicates: {values}.")
    return values


def align_node_attrs(
    dataset,
    target_atomic_numbers: Sequence[int] | torch.Tensor,
):
    """Align graph node attributes with the target atomic-number ordering."""
    if not hasattr(dataset, "metadata"):
        raise TypeError("The dataset must expose a `metadata` attribute.")
    if "atomic_numbers" not in dataset.metadata:
        raise KeyError("The dataset metadata must contain `atomic_numbers`.")

    source = _as_atomic_number_list(dataset.metadata["atomic_numbers"], "source")
    target = _as_atomic_number_list(target_atomic_numbers, "target")

    if source == target:
        return dataset

    missing = set(source) - set(target)
    if missing:
        raise ValueError(f"Unsupported atomic numbers: {sorted(missing)}.")

    indices = [target.index(z) for z in source]

    for graph in dataset["data_list"]:
        attrs = graph["node_attrs"]
        if attrs.ndim != 2 or attrs.shape[1] != len(source):
            raise ValueError("Invalid `node_attrs` dimensions.")

        aligned = attrs.new_zeros(len(attrs), len(target))
        aligned[:, indices] = attrs
        graph["node_attrs"] = aligned

    dataset.metadata["atomic_numbers"] = target
    return dataset
