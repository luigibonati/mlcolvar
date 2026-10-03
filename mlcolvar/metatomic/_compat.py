from contextlib import contextmanager

import torch


@contextmanager
def _torch_indexing_compat():
    """Temporarily restore PyTorch indexing patched by PyG HashTensor."""
    try:
        import torch_geometric.hash_tensor as hash_tensor
    except ImportError:
        yield
        return

    index_select = torch.index_select
    select = torch.select
    torch.index_select = getattr(hash_tensor, "_old_index_select", index_select)
    torch.select = getattr(hash_tensor, "_old_select", select)

    try:
        yield
    finally:
        torch.index_select = index_select
        torch.select = select


with _torch_indexing_compat():
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