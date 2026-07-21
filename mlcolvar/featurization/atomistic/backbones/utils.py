from __future__ import annotations

from typing import List

import torch


def to_int(
    value,
    name: str,
) -> int:
    """Convert a scalar tensor or Python scalar to an integer."""

    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError(
                f"{name} must contain exactly one value."
            )

        return int(
            value.detach().cpu().item()
        )

    return int(value)


def to_float(
    value,
    name: str,
) -> float:
    """Convert a scalar tensor or Python scalar to a float."""

    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError(
                f"{name} must contain exactly one value."
            )

        return float(
            value.detach().cpu().item()
        )

    return float(value)


def to_bool(
    value,
    name: str,
) -> bool:
    """Convert a scalar tensor or Python scalar to a boolean."""

    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError(
                f"{name} must contain exactly one value."
            )

        return bool(
            value.detach().cpu().item()
        )

    return bool(value)


def to_int_list(
    value,
    name: str,
    *,
    allow_empty: bool = False,
) -> List[int]:
    """Convert a tensor or sequence to a list of integers."""

    if isinstance(value, torch.Tensor):
        values = (
            value.detach()
            .cpu()
            .reshape(-1)
            .tolist()
        )

    else:
        values = list(value)

    result = [
        int(item)
        for item in values
    ]

    if (
        not allow_empty
        and len(result) == 0
    ):
        raise ValueError(
            f"{name} must not be empty."
        )

    return result
