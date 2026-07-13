import sys

import torch

_ALREADY_IMPORTED = (
    "metatensor_torch.operations._dispatch" in sys.modules
    or "metatensor.torch.operations._dispatch" in sys.modules
    or "metatrain.pet.model" in sys.modules
)

if _ALREADY_IMPORTED:
    raise RuntimeError(
        "PET/metatensor was imported before the TorchScript compatibility "
        "prelude. Restart Python and import pet_torchscript_prelude first."
    )

try:
    import torch_geometric.hash_tensor as _hash_tensor
except ImportError:
    _hash_tensor = None

if _hash_tensor is not None:
    _native_index_select = getattr(
        _hash_tensor,
        "_old_index_select",
        None,
    )
    _native_select = getattr(
        _hash_tensor,
        "_old_select",
        None,
    )

    if _native_index_select is not None:
        torch.index_select = _native_index_select

    if _native_select is not None:
        torch.select = _native_select
