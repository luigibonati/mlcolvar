"""Optional Metatomic imports and Torch indexing compatibility."""

import torch


def _restore_native_torch_indexing() -> None:
    """Undo the global PyG HashTensor indexing monkey-patch."""
    try:
        import torch_geometric.hash_tensor as hash_tensor
    except ImportError:
        return

    torch.index_select = getattr(
        hash_tensor,
        "_old_index_select",
        torch.index_select,
    )
    torch.select = getattr(
        hash_tensor,
        "_old_select",
        torch.select,
    )


# This must run before importing metatensor/metatomic.
_restore_native_torch_indexing()

try:
    from metatensor.torch import Labels, TensorBlock, TensorMap
    from metatomic.torch import (
        AtomisticModel as MetatomicAtomisticModel,
        ModelCapabilities,
        ModelMetadata,
        ModelOutput,
        NeighborListOptions,
        System,
    )
except ImportError as exc:
    raise ImportError(
        "Atomistic Metatomic export requires both 'metatensor-torch' and "
        "'metatomic-torch'. Install these optional dependencies before "
        "importing mlcolvar.integrations.atomistic.metatomic."
    ) from exc


__all__ = [
    "Labels",
    "TensorBlock",
    "TensorMap",
    "MetatomicAtomisticModel",
    "ModelCapabilities",
    "ModelMetadata",
    "ModelOutput",
    "NeighborListOptions",
    "System",
]
