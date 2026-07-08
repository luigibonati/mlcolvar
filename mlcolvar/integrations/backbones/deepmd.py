from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import nn

from mlcolvar.integrations.atomistic import BaseAtomisticBackbone
from mlcolvar.integrations.utils import (
    get_graph_ptr,
    prepare_cells,
    prepare_pbc,
)

from ._utils import to_bool, to_float, to_int, to_int_list


try:
    from ase.data import atomic_numbers as ase_atomic_numbers

    from deepmd.pt.utils import env as deepmd_env
    from deepmd.pt.utils.nlist import extend_input_and_build_neighbor_list

    _DEEPMD_AVAILABLE = True
    _DEEPMD_IMPORT_ERROR = None

except ImportError as exc:
    ase_atomic_numbers = None
    deepmd_env = None
    extend_input_and_build_neighbor_list = None

    _DEEPMD_AVAILABLE = False
    _DEEPMD_IMPORT_ERROR = exc


__all__ = ["DeepMDBackbone"]


_PRECISION_TO_DTYPE = {
    "float16": torch.float16,
    "half": torch.float16,
    "float32": torch.float32,
    "single": torch.float32,
    "float64": torch.float64,
    "double": torch.float64,
    "bfloat16": torch.bfloat16,
}

_DTYPE_TO_PRECISION = {
    torch.float16: "float16",
    torch.float32: "float32",
    torch.float64: "float64",
    torch.bfloat16: "bfloat16",
}


def _normalize_precision(
    value: Any,
    *,
    name: str,
    allow_default: bool = False,
) -> Optional[Tuple[str, torch.dtype]]:
    """Normalize a precision declaration."""

    if isinstance(value, torch.dtype):
        dtype = value
    else:
        value = str(value).strip().lower().replace("torch.", "")

        if value == "default" and allow_default:
            return None

        if value not in _PRECISION_TO_DTYPE:
            raise ValueError(f"Unsupported {name}: {value!r}.")

        dtype = _PRECISION_TO_DTYPE[value]

    if dtype not in _DTYPE_TO_PRECISION:
        raise ValueError(f"Unsupported {name}: {dtype}.")

    return _DTYPE_TO_PRECISION[dtype], dtype


def _infer_module_precision(
    module: nn.Module,
    *,
    name: str,
) -> Tuple[str, torch.dtype]:
    """Infer precision from floating-point parameters or buffers."""

    for source_name, tensors in (
        ("parameters", module.parameters()),
        ("buffers", module.buffers()),
    ):
        dtypes = {
            tensor.dtype
            for tensor in tensors
            if tensor.is_floating_point()
        }

        if len(dtypes) > 1:
            raise ValueError(
                f"The {name} contains {source_name} with mixed "
                f"floating-point precisions: "
                f"{sorted(str(dtype) for dtype in dtypes)}."
            )

        if dtypes:
            result = _normalize_precision(
                next(iter(dtypes)),
                name=f"{name} {source_name} precision",
            )
            assert result is not None
            return result

    raise ValueError(
        f"Could not infer the floating-point precision of the {name}."
    )


def _infer_descriptor_precision(
    descriptor: nn.Module,
) -> Tuple[str, torch.dtype]:
    """Infer DeePMD descriptor precision."""

    for attribute in ("prec", "precision"):
        value = getattr(descriptor, attribute, None)

        if value is None:
            continue

        result = _normalize_precision(
            value,
            name=f"descriptor.{attribute}",
            allow_default=True,
        )

        if result is not None:
            return result

    return _infer_module_precision(
        descriptor,
        name="DeePMD descriptor",
    )


def _infer_neighbor_list_precision(
    descriptor_dtype: torch.dtype,
) -> Tuple[str, torch.dtype]:
    """Infer DeePMD neighbor-list precision."""

    result = _normalize_precision(
        getattr(
            deepmd_env,
            "GLOBAL_PT_FLOAT_PRECISION",
            descriptor_dtype,
        ),
        name="DeePMD global neighbor-list precision",
        allow_default=True,
    )

    if result is not None:
        return result

    fallback = _normalize_precision(
        descriptor_dtype,
        name="descriptor precision",
    )
    assert fallback is not None

    return fallback


def _is_native_deepmd_model(
    model: nn.Module,
) -> bool:
    """Check for a native DeePMD PyTorch model."""

    if not all(
        callable(getattr(model, name, None))
        for name in ("get_descriptor", "get_type_map")
    ):
        return False

    try:
        descriptor = model.get_descriptor()
    except Exception:
        return False

    return all(
        callable(getattr(descriptor, name, None))
        for name in (
            "get_rcut",
            "get_sel",
            "get_dim_out",
            "mixed_types",
        )
    )


def _unwrap_deepmd_model(
    model: nn.Module,
) -> nn.Module:
    """Extract a native DeePMD PyTorch model from common wrappers."""

    if _is_native_deepmd_model(model):
        return model

    for attribute in ("module", "model"):
        candidate = getattr(model, attribute, None)

        if isinstance(candidate, nn.Module) and _is_native_deepmd_model(
            candidate
        ):
            return candidate

    raise ValueError(
        "Could not find a native DeePMD-kit PyTorch model. The supplied "
        "module must expose `get_descriptor()` and `get_type_map()`."
    )


def _get_type_map(
    model: nn.Module,
    descriptor: nn.Module,
) -> List[str]:
    """Return the DeePMD element type map."""

    for module in (model, descriptor):
        getter = getattr(module, "get_type_map", None)

        if callable(getter):
            type_map = getter()

            if type_map is not None and len(type_map) > 0:
                return list(type_map)

    raise ValueError(
        "The DeePMD model does not provide a non-empty element type map."
    )


def _type_map_to_atomic_numbers(
    type_map: List[str],
) -> List[int]:
    """Convert a DeePMD type map to atomic numbers."""

    atomic_numbers: List[int] = []

    for symbol in type_map:
        if not isinstance(symbol, str) or symbol not in ase_atomic_numbers:
            raise ValueError(
                "Invalid DeePMD element symbol in type map: "
                f"{symbol!r}."
            )

        atomic_numbers.append(int(ase_atomic_numbers[symbol]))

    if len(set(atomic_numbers)) != len(atomic_numbers):
        raise ValueError("The DeePMD type map contains duplicate elements.")

    return atomic_numbers


class DeepMDBackbone(BaseAtomisticBackbone):
    """Extract atom-level descriptors from a DeePMD PyTorch model."""

    __constants__ = [
        "descriptor_cutoff",
        "descriptor_dim",
        "descriptor_precision",
        "neighbor_list_precision",
        "mixed_types",
    ]

    def __init__(
        self,
        model: nn.Module,
        buffer: float = 0.0,
        long_range_cutoff: float = -1.0,
    ) -> None:
        if not _DEEPMD_AVAILABLE:
            raise ImportError(
                "DeepMDBackbone requires DeePMD-kit with the PyTorch backend."
            ) from _DEEPMD_IMPORT_ERROR

        if not isinstance(model, nn.Module):
            raise TypeError("`model` must be a torch.nn.Module.")

        if long_range_cutoff >= 0.0:
            raise ValueError(
                "DeepMDBackbone does not support `long_range_cutoff`."
            )

        model = _unwrap_deepmd_model(model)
        descriptor = model.get_descriptor()

        type_map = _get_type_map(
            model=model,
            descriptor=descriptor,
        )

        atomic_numbers = _type_map_to_atomic_numbers(type_map)

        descriptor_cutoff = to_float(
            descriptor.get_rcut(),
            name="descriptor.get_rcut()",
        )

        descriptor_dim = to_int(
            descriptor.get_dim_out(),
            name="descriptor.get_dim_out()",
        )

        selection = to_int_list(
            descriptor.get_sel(),
            name="descriptor.get_sel()",
        )

        mixed_types = to_bool(
            descriptor.mixed_types(),
            name="descriptor.mixed_types()",
        )

        descriptor_precision, descriptor_dtype = _infer_descriptor_precision(
            descriptor
        )

        neighbor_list_precision, neighbor_list_dtype = (
            _infer_neighbor_list_precision(descriptor_dtype)
        )

        if descriptor_cutoff <= 0.0:
            raise ValueError(
                "The DeePMD descriptor cutoff must be positive, "
                f"found {descriptor_cutoff}."
            )

        if descriptor_dim <= 0:
            raise ValueError(
                "The DeePMD descriptor output dimension must be positive, "
                f"found {descriptor_dim}."
            )

        if any(value < 0 for value in selection):
            raise ValueError(
                "DeePMD neighbor selection values must be non-negative, "
                f"found {selection}."
            )

        if not mixed_types and len(selection) != len(atomic_numbers):
            raise ValueError(
                "`descriptor.get_sel()` must contain one entry per element "
                "type when `mixed_types=False`."
            )

        super().__init__(
            out_features=descriptor_dim,
            atomic_numbers=atomic_numbers,
            cutoff=descriptor_cutoff,
            sample_kind="atom",
            buffer=buffer,
            long_range_cutoff=-1.0,
            full_neighbor_list=False,
        )

        self.descriptor_cutoff = descriptor_cutoff
        self.descriptor_dim = descriptor_dim
        self.descriptor_precision = descriptor_precision
        self.neighbor_list_precision = neighbor_list_precision
        self.selection: List[int] = selection
        self.mixed_types = mixed_types
        self.type_map: List[str] = type_map

        self.register_buffer(
            "_descriptor_dtype_reference",
            torch.zeros((), dtype=descriptor_dtype),
        )

        self.register_buffer(
            "_neighbor_list_dtype_reference",
            torch.zeros((), dtype=neighbor_list_dtype),
        )

        # Register only the descriptor. The full DeePMD potential is not
        # needed after metadata has been collected.
        self.descriptor = descriptor
        self._restore_internal_precision()

    @staticmethod
    @torch.jit.unused
    def _find_module_device(
        module: nn.Module,
    ) -> Optional[torch.device]:
        """Return the first parameter/buffer device."""

        for tensor in module.parameters():
            return tensor.device

        for tensor in module.buffers():
            return tensor.device

        return None

    @torch.jit.unused
    def _restore_internal_precision(self) -> None:
        """Restore fixed DeePMD internal precisions."""

        if not hasattr(self, "descriptor"):
            return

        descriptor_dtype = _PRECISION_TO_DTYPE[self.descriptor_precision]
        neighbor_list_dtype = _PRECISION_TO_DTYPE[
            self.neighbor_list_precision
        ]

        self.descriptor.to(dtype=descriptor_dtype)

        descriptor_device = self._find_module_device(self.descriptor)

        if descriptor_device is None:
            descriptor_device = self._descriptor_dtype_reference.device

        self._descriptor_dtype_reference = self._descriptor_dtype_reference.to(
            device=descriptor_device,
            dtype=descriptor_dtype,
        )

        self._neighbor_list_dtype_reference = (
            self._neighbor_list_dtype_reference.to(
                device=descriptor_device,
                dtype=neighbor_list_dtype,
            )
        )

    def _apply(
        self,
        fn,
        recurse: bool = True,
    ):
        """Apply transforms while preserving DeePMD internal precision."""

        module = super()._apply(
            fn,
            recurse=recurse,
        )

        if hasattr(self, "descriptor"):
            self._restore_internal_precision()

        return module

    def forward(
        self,
        data: Dict[str, torch.Tensor],
        cell: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute atom-level DeePMD descriptors."""

        input_positions = data["positions"]
        output_dtype = input_positions.dtype
        output_device = input_positions.device

        descriptor_dtype = self._descriptor_dtype_reference.dtype
        descriptor_device = self._descriptor_dtype_reference.device
        neighbor_list_dtype = self._neighbor_list_dtype_reference.dtype

        positions = input_positions.to(
            device=descriptor_device,
            dtype=descriptor_dtype,
        )

        ptr = get_graph_ptr(
            data=data,
            n_atoms=positions.size(0),
            device=descriptor_device,
        )

        n_systems = ptr.numel() - 1

        cells = prepare_cells(
            data=data,
            cell=cell,
            n_systems=n_systems,
            positions=positions,
        )

        pbc = prepare_pbc(
            data=data,
            cells=cells,
            n_systems=n_systems,
        )

        atom_types = data["node_attrs"].argmax(dim=-1).to(
            device=descriptor_device,
            dtype=torch.long,
        )

        feature_blocks = torch.jit.annotate(
            List[torch.Tensor],
            [],
        )

        for system_index in range(n_systems):
            start = int(ptr[system_index].item())
            end = int(ptr[system_index + 1].item())

            features = self._forward_system(
                positions=positions[start:end],
                atom_types=atom_types[start:end],
                cell=cells[system_index],
                pbc=pbc[system_index],
                neighbor_list_dtype=neighbor_list_dtype,
                descriptor_dtype=descriptor_dtype,
            )

            feature_blocks.append(features)

        if len(feature_blocks) == 0:
            return input_positions.new_empty((0, self.out_features))

        output = torch.cat(
            feature_blocks,
            dim=0,
        )

        return output.to(
            device=output_device,
            dtype=output_dtype,
        )

    def _forward_system(
        self,
        positions: torch.Tensor,
        atom_types: torch.Tensor,
        cell: torch.Tensor,
        pbc: torch.Tensor,
        neighbor_list_dtype: torch.dtype,
        descriptor_dtype: torch.dtype,
    ) -> torch.Tensor:
        """Compute descriptors for one system."""

        positions = positions.unsqueeze(0)
        atom_types = atom_types.unsqueeze(0)

        box = torch.jit.annotate(
            Optional[torch.Tensor],
            None,
        )

        if bool(torch.all(pbc).item()):
            box = cell.unsqueeze(0)
        elif bool(torch.any(pbc).item()):
            raise ValueError(
                "DeepMDBackbone supports only fully periodic or fully "
                "non-periodic systems."
            )

        neighbor_box = torch.jit.annotate(
            Optional[torch.Tensor],
            None,
        )

        if box is not None:
            neighbor_box = box.to(dtype=neighbor_list_dtype)

        (
            extended_coord,
            extended_atype,
            mapping,
            neighbor_list,
        ) = extend_input_and_build_neighbor_list(
            coord=positions.to(dtype=neighbor_list_dtype),
            atype=atom_types,
            rcut=self.descriptor_cutoff,
            sel=self.selection,
            mixed_types=self.mixed_types,
            box=neighbor_box,
        )

        output = self.descriptor(
            extended_coord.to(dtype=descriptor_dtype),
            extended_atype,
            neighbor_list,
            mapping,
        )

        return output[0][0]