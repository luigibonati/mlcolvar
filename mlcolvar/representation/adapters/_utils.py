from typing import Any

import torch


def to_int(value: Any, *, name: str) -> int:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError(f"`{name}` must be scalar.")
        value = value.detach().cpu().item()
    return int(value)


def to_float(value: Any, *, name: str) -> float:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError(f"`{name}` must be scalar.")
        value = value.detach().cpu().item()
    return float(value)


def to_bool(value: Any, *, name: str) -> bool:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError(f"`{name}` must be scalar.")
        value = value.detach().cpu().item()
    return bool(value)


def to_int_list(value: Any, *, name: str) -> list[int]:
    try:
        values = torch.as_tensor(value, dtype=torch.long).detach().cpu().reshape(-1)
    except Exception as exc:
        raise ValueError(f"Could not convert `{name}` to an integer list.") from exc
    return [int(item) for item in values.tolist()]
