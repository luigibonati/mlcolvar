import torch


def _restore_torch_indexing() -> None:
    """Restore PyTorch indexing potentially patched by PyG HashTensor."""
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


_restore_torch_indexing()

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
        "Metatomic export requires 'metatensor-torch' and 'metatomic-torch'."
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