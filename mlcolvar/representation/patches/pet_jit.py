from __future__ import annotations

import sys

import torch


__all__ = [
    "ensure_pet_jit_compatibility",
]


_PATCHED = False


def _incompatible_modules_loaded() -> list[str]:
    """Return modules that make the PET TorchScript patch too late."""

    candidates = (
        "metatensor_torch.operations._dispatch",
        "metatensor.torch.operations._dispatch",
        "metatrain.pet.model",
    )

    return [
        name
        for name in candidates
        if name in sys.modules
    ]


def ensure_pet_jit_compatibility() -> None:
    """Install the PET TorchScript compatibility patch.

    This function is intentionally explicit. Importing PET representations
    for normal eager execution must not impose any global import-order
    requirement.

    The patch only needs to be installed before PET TorchScript/export
    functionality is used.
    """

    global _PATCHED

    if _PATCHED:
        return

    already_loaded = (
        _incompatible_modules_loaded()
    )

    if already_loaded:
        raise RuntimeError(
            "PET TorchScript compatibility must be installed "
            "before importing the relevant metatensor/PET modules. "
            "Already imported: "
            + ", ".join(already_loaded)
        )

    try:
        import torch_geometric.hash_tensor as hash_tensor
    except ImportError:
        # Nothing to patch when torch_geometric's HashTensor
        # compatibility layer is unavailable.
        _PATCHED = True
        return

    native_index_select = getattr(
        hash_tensor,
        "_old_index_select",
        None,
    )

    native_select = getattr(
        hash_tensor,
        "_old_select",
        None,
    )

    if native_index_select is not None:
        torch.index_select = native_index_select

    if native_select is not None:
        torch.select = native_select

    _PATCHED = True