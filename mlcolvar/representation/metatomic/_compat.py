import torch


def _restore_native_torch_indexing() -> None:
    """Restore native PyTorch indexing functions before importing Metatomic.

    PyTorch Geometric may replace ``torch.index_select`` and ``torch.select``
    when HashTensor support is imported. Metatensor/Metatomic expects the
    native PyTorch implementations, so restore them before importing the
    optional Metatomic dependencies.
    """
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
        "importing mlcolvar.representation.metatomic."
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