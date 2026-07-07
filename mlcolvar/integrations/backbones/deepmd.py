from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import nn

from mlcolvar.integrations.atomistic import BaseAtomisticBackbone

from ._utils import to_bool, to_float, to_int, to_int_list


try:
    from ase.data import atomic_numbers as ase_atomic_numbers

    from deepmd.pt.utils import env as deepmd_env
    from deepmd.pt.utils.nlist import extend_input_and_build_neighbor_list

    _DEEPMD_AVAILABLE = True
    _DEEPMD_IMPORT_ERROR = None
except ImportError as exc:
    # Keep mlcolvar importable when the optional DeePMD dependencies are
    # unavailable.
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
    """Normalize a precision declaration to a canonical name and dtype."""
    if isinstance(value, torch.dtype):
        dtype = value
    else:
        normalized = str(value).strip().lower().replace("torch.", "")

        if normalized == "default" and allow_default:
            return None

        if normalized not in _PRECISION_TO_DTYPE:
            raise ValueError(f"Unsupported {name}: {value!r}.")

        dtype = _PRECISION_TO_DTYPE[normalized]

    if dtype not in _DTYPE_TO_PRECISION:
        raise ValueError(f"Unsupported {name}: {dtype}.")

    return _DTYPE_TO_PRECISION[dtype], dtype


def _type_map_to_atomic_numbers(type_map: List[str]) -> List[int]:
    """Convert a DeePMD element type map to atomic numbers."""
    atomic_number_list: List[int] = []

    for symbol in type_map:
        if not isinstance(symbol, str) or symbol not in ase_atomic_numbers:
            raise ValueError(
                "DeepMDBackbone requires every entry in the DeePMD type map "
                "to be a valid chemical-element symbol. "
                f"Could not interpret {symbol!r}."
            )

        atomic_number_list.append(int(ase_atomic_numbers[symbol]))

    if len(set(atomic_number_list)) != len(atomic_number_list):
        raise ValueError("The DeePMD type map contains duplicate elements.")

    return atomic_number_list


def _is_native_deepmd_model(model: nn.Module) -> bool:
    """Check for the native DeePMD PyTorch descriptor interface."""
    required_methods = ("get_descriptor", "get_type_map")

    if not all(callable(getattr(model, name, None)) for name in required_methods):
        return False

    try:
        descriptor = model.get_descriptor()
    except Exception:
        # Interface probing should not propagate errors from incompatible
        # wrappers.
        return False

    descriptor_methods = (
        "get_rcut",
        "get_sel",
        "get_dim_out",
        "mixed_types",
    )

    return all(
        callable(getattr(descriptor, name, None)) for name in descriptor_methods
    )


def _unwrap_deepmd_model(model: nn.Module) -> nn.Module:
    """Extract a native DeePMD PyTorch model from common wrappers."""
    if _is_native_deepmd_model(model):
        return model

    for attribute in ("module", "model"):
        candidate = getattr(model, attribute, None)
        if isinstance(candidate, nn.Module) and _is_native_deepmd_model(candidate):
            return candidate

    raise ValueError(
        "Could not find a native DeePMD-kit PyTorch model. The supplied "
        "module must expose `get_descriptor()` and `get_type_map()`. "
        "TensorFlow frozen models and the NumPy-based DeepPotential "
        "interface are not supported."
    )


def _infer_module_precision(
    module: nn.Module,
    *,
    name: str,
) -> Tuple[str, torch.dtype]:
    """Infer a module precision from parameters, then buffers."""
    parameter_dtypes = {
        parameter.dtype
        for parameter in module.parameters()
        if parameter.is_floating_point()
    }

    if len(parameter_dtypes) > 1:
        raise ValueError(
            f"The {name} contains parameters with mixed floating-point "
            f"precisions: {sorted(str(dtype) for dtype in parameter_dtypes)}."
        )

    if parameter_dtypes:
        result = _normalize_precision(
            next(iter(parameter_dtypes)),
            name=f"{name} parameter precision",
        )
        assert result is not None
        return result

    buffer_dtypes = {
        buffer.dtype for buffer in module.buffers() if buffer.is_floating_point()
    }

    if len(buffer_dtypes) > 1:
        raise ValueError(
            f"The {name} contains buffers with mixed floating-point "
            f"precisions: {sorted(str(dtype) for dtype in buffer_dtypes)}."
        )

    if buffer_dtypes:
        result = _normalize_precision(
            next(iter(buffer_dtypes)),
            name=f"{name} buffer precision",
        )
        assert result is not None
        return result

    raise ValueError(f"Could not infer the floating-point precision of the {name}.")


def _infer_descriptor_precision(
    descriptor: nn.Module,
) -> Tuple[str, torch.dtype]:
    """Infer the floating-point precision used by a DeePMD descriptor."""
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

    return _infer_module_precision(descriptor, name="DeePMD descriptor")


def _infer_neighbor_list_precision(
    descriptor_dtype: torch.dtype,
) -> Tuple[str, torch.dtype]:
    """Infer DeePMD's global neighbor-list construction precision."""
    value = getattr(
        deepmd_env,
        "GLOBAL_PT_FLOAT_PRECISION",
        descriptor_dtype,
    )

    result = _normalize_precision(
        value,
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



class DeepMDBackbone(BaseAtomisticBackbone):
    """Extract atom-level descriptors from a DeePMD PyTorch model.

    This adapter targets native PyTorch DeePMD models, including DPA-2.
    DeePMD constructs its own padded neighbor list from coordinates, atom
    types, and the simulation cell, so the mlcolvar ``edge_index`` is not
    used by this backbone.

    DeePMD's descriptor and neighbor-list implementation can use different
    floating-point precisions:

    - descriptor calculations follow the descriptor precision;
    - periodic ghost-atom construction follows
      ``deepmd_env.GLOBAL_PT_FLOAT_PRECISION``;
    - returned features use the dtype and device of the input graph.

    These dtype boundaries are preserved without breaking the autograd path
    from output features to the original input coordinates.

    Parameters
    ----------
    model
        Native DeePMD-kit PyTorch model exposing ``get_descriptor()`` and
        ``get_type_map()``. Common wrappers exposing the native model through
        ``.module`` or ``.model`` are also accepted.
    buffer
        Retained for compatibility with the common atomistic-backbone
        interface. DeePMD constructs its own neighbor list.
    long_range_cutoff
        Unsupported because DeePMD constructs its own neighbor list. This
        value must remain negative.
    """

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
                "DeepMDBackbone requires DeePMD-kit with the PyTorch "
                "backend. Install a DeePMD-kit build that matches the "
                "installed PyTorch version."
            ) from _DEEPMD_IMPORT_ERROR

        if not isinstance(model, nn.Module):
            raise TypeError("`model` must be a torch.nn.Module.")

        if long_range_cutoff >= 0.0:
            raise ValueError(
                "DeepMDBackbone does not support `long_range_cutoff` because "
                "DeePMD constructs its own neighbor list."
            )

        model = _unwrap_deepmd_model(model)
        descriptor = model.get_descriptor()

        raw_type_map = model.get_type_map()
        type_map: List[str] = [] if raw_type_map is None else list(raw_type_map)

        if len(type_map) == 0:
            descriptor_type_map = getattr(descriptor, "get_type_map", None)
            if callable(descriptor_type_map):
                raw_descriptor_type_map = descriptor_type_map()
                if raw_descriptor_type_map is not None:
                    type_map = list(raw_descriptor_type_map)

        if len(type_map) == 0:
            raise ValueError(
                "The DeePMD model does not provide a non-empty element type map."
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
                "For a type-distinguished DeePMD neighbor list, "
                "`descriptor.get_sel()` must contain one entry per element "
                f"type. Found {len(selection)} selection entries and "
                f"{len(atomic_numbers)} element types."
            )

        super().__init__(
            out_features=descriptor_dim,
            atomic_numbers=atomic_numbers,
            cutoff=descriptor_cutoff,
            sample_kind="atom",
            buffer=buffer,
            long_range_cutoff=-1.0,
            # DeePMD ignores the mlcolvar edge list and constructs its own
            # padded neighbor list.
            full_neighbor_list=False,
        )

        self.descriptor_cutoff = descriptor_cutoff
        self.descriptor_dim = descriptor_dim
        self.descriptor_precision = descriptor_precision
        self.neighbor_list_precision = neighbor_list_precision
        self.selection: List[int] = selection
        self.mixed_types = mixed_types
        self.type_map: List[str] = type_map

        # Scalar buffers track DeePMD's internal device and fixed floating-
        # point precisions. Scalar tensors are convenient TorchScript
        # attributes and are serialized together with the scripted module.
        self.register_buffer(
            "_descriptor_dtype_reference",
            torch.zeros((), dtype=descriptor_dtype),
        )
        self.register_buffer(
            "_neighbor_list_dtype_reference",
            torch.zeros((), dtype=neighbor_list_dtype),
        )

        # Register the descriptor itself as a direct child module. Calling
        # ``model.get_descriptor()`` inside forward would return a module
        # through a regular Python method; TorchScript does not treat such a
        # dynamically returned object as a callable submodule.
        #
        # The complete DeePMD potential is not needed for CV inference after
        # the descriptor metadata has been collected above.
        self.descriptor = descriptor
        self._restore_internal_precision()

    @staticmethod
    @torch.jit.unused
    def _find_module_device(module: nn.Module) -> Optional[torch.device]:
        """Return the device of the first parameter or buffer in a module."""
        for parameter in module.parameters():
            return parameter.device
        for buffer in module.buffers():
            return buffer.device
        return None

    @torch.jit.unused
    def _restore_internal_precision(self) -> None:
        """Restore DeePMD's fixed internal precisions after transforms."""
        if not hasattr(self, "descriptor"):
            return

        descriptor_dtype = _PRECISION_TO_DTYPE[self.descriptor_precision]
        neighbor_list_dtype = _PRECISION_TO_DTYPE[
            self.neighbor_list_precision
        ]

        # A dtype-only conversion preserves device changes performed by a
        # surrounding ``to(device=...)`` call.
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

    def _apply(self, fn, recurse: bool = True):
        """Apply transforms while preserving DeePMD internal precision.

        Device changes requested through ``to(device=...)`` are retained.
        Calls such as ``float()``, ``double()``, and ``to(dtype=...)`` do
        not change the descriptor or neighbor-list construction precision.
        """
        module = super()._apply(fn, recurse=recurse)

        if hasattr(self, "descriptor"):
            self._restore_internal_precision()

        return module

    def forward(
        self,
        data: Dict[str, torch.Tensor],
        cell: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute atom-level DeePMD descriptors.

        Parameters
        ----------
        data
            Batched mlcolvar graph containing at least ``positions`` and
            ``node_attrs``. ``ptr`` or ``batch`` can be supplied to separate
            systems.
        cell
            Optional runtime cell. When supplied, it overrides
            ``data["cell"]``.

        Returns
        -------
        torch.Tensor
            Atom-level descriptors with shape ``[n_atoms, descriptor_dim]``
            using the dtype and device of the input positions.
        """
        if not torch.jit.is_scripting() and not torch.jit.is_tracing():
            self._validate_graph_input(data=data, cell=cell)

        input_positions = data["positions"]
        output_dtype = input_positions.dtype
        output_device = input_positions.device

        descriptor_dtype = self._descriptor_dtype_reference.dtype
        descriptor_device = self._descriptor_dtype_reference.device
        neighbor_list_dtype = self._neighbor_list_dtype_reference.dtype

        # Moving and casting preserve the autograd connection to the original
        # input coordinates.
        positions = input_positions.to(
            device=descriptor_device,
            dtype=descriptor_dtype,
        )

        ptr = self._get_ptr(
            data=data,
            n_atoms=positions.size(0),
            device=descriptor_device,
        )
        n_systems = ptr.numel() - 1

        cells = self._prepare_cells(
            data=data,
            cell=cell,
            n_systems=n_systems,
            positions=positions,
        )
        pbc = self._prepare_pbc(
            data=data,
            cells=cells,
            n_systems=n_systems,
        )

        # align_dataset() arranges node_attrs in DeePMD type-map order.
        atom_types = data["node_attrs"].argmax(dim=-1).to(
            device=descriptor_device,
            dtype=torch.long,
        )

        feature_blocks = torch.jit.annotate(List[torch.Tensor], [])

        for system_index in range(n_systems):
            start = int(ptr[system_index].item())
            end = int(ptr[system_index + 1].item())

            system_positions = positions[start:end].unsqueeze(0)
            system_types = atom_types[start:end].unsqueeze(0)
            system_pbc = pbc[system_index]

            if bool(torch.all(system_pbc).item()):
                box: Optional[torch.Tensor] = cells[system_index].unsqueeze(0)
            elif bool(torch.any(system_pbc).item()):
                raise ValueError(
                    "DeepMDBackbone currently supports only fully periodic "
                    "or fully non-periodic systems. Partial periodic boundary "
                    "conditions are unsupported."
                )
            else:
                box = None

            # Periodic ghost-atom construction follows DeePMD's global
            # neighbor-list precision.
            neighbor_positions = system_positions.to(dtype=neighbor_list_dtype)
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
                coord=neighbor_positions,
                atype=system_types,
                rcut=self.descriptor_cutoff,
                sel=self.selection,
                mixed_types=self.mixed_types,
                box=neighbor_box,
            )

            # Descriptor calculations can use a different precision from the
            # neighbor-list construction.
            extended_coord = extended_coord.to(dtype=descriptor_dtype)

            # ``self.descriptor`` is a registered child module, so
            # TorchScript can compile this call. DPA descriptors return the
            # atom-level descriptor as the first item of their output tuple.
            descriptor_output = self.descriptor(
                extended_coord,
                extended_atype,
                neighbor_list,
                mapping,
            )
            features = descriptor_output[0]

            if not torch.jit.is_scripting() and not torch.jit.is_tracing():
                self._validate_descriptor_output(
                    features=features,
                    n_atoms=end - start,
                )

            feature_blocks.append(features[0])

        if len(feature_blocks) == 0:
            return input_positions.new_empty((0, self.out_features))

        output = torch.cat(feature_blocks, dim=0)

        if not torch.jit.is_scripting() and not torch.jit.is_tracing():
            self._validate_combined_output(
                output=output,
                n_atoms=positions.size(0),
            )

        # Restore the dtype and device expected by the surrounding mlcolvar
        # graph. This cast also preserves coordinate gradients.
        return output.to(device=output_device, dtype=output_dtype)

    @torch.jit.unused
    def _validate_descriptor_output(
        self,
        features: torch.Tensor,
        n_atoms: int,
    ) -> None:
        """Validate one system's raw DeePMD descriptor output."""
        if features.dim() != 3:
            raise RuntimeError(
                "Expected DeePMD descriptor output with shape "
                "[n_frames, n_atoms, n_features], but found "
                f"{tuple(features.shape)}."
            )

        if features.size(0) != 1:
            raise RuntimeError(
                "DeepMDBackbone processes one system at a time and expected "
                "exactly one descriptor frame."
            )

        if features.size(1) != n_atoms:
            raise RuntimeError(
                "The number of DeePMD descriptor rows does not match the "
                f"number of local atoms. Expected {n_atoms}, found "
                f"{features.size(1)}."
            )

        if features.size(2) != self.out_features:
            raise RuntimeError(
                "Unexpected DeePMD descriptor dimension. Expected "
                f"{self.out_features}, found {features.size(2)}."
            )

    @torch.jit.unused
    def _validate_combined_output(
        self,
        output: torch.Tensor,
        n_atoms: int,
    ) -> None:
        """Validate the combined atom-level descriptor tensor."""
        if output.dim() != 2:
            raise RuntimeError(
                "Expected the combined DeePMD output to have shape "
                f"[n_atoms, n_features], but found {tuple(output.shape)}."
            )

        if output.size(0) != n_atoms:
            raise RuntimeError(
                "The total number of DeePMD descriptor rows does not match "
                f"the graph atom count. Expected {n_atoms}, found "
                f"{output.size(0)}."
            )

        if output.size(1) != self.out_features:
            raise RuntimeError(
                "Unexpected DeePMD descriptor dimension. Expected "
                f"{self.out_features}, found {output.size(1)}."
            )

    @torch.jit.unused
    def _validate_graph_input(
        self,
        data: Dict[str, torch.Tensor],
        cell: Optional[torch.Tensor],
    ) -> None:
        """Validate graph fields required by the DeePMD adapter."""
        required_keys = ("positions", "node_attrs")
        missing_keys = [key for key in required_keys if key not in data]

        if missing_keys:
            raise KeyError(
                "DeepMD graph input is missing required fields: "
                f"{missing_keys}."
            )

        positions = data["positions"]
        node_attrs = data["node_attrs"]

        if positions.dim() != 2 or positions.size(1) != 3:
            raise ValueError(
                "`positions` must have shape [n_atoms, 3], "
                f"found {tuple(positions.shape)}."
            )

        if not positions.is_floating_point():
            raise TypeError("`positions` must use a floating-point dtype.")

        if node_attrs.dim() != 2:
            raise ValueError(
                "`node_attrs` must be a rank-2 tensor, "
                f"found shape {tuple(node_attrs.shape)}."
            )

        if node_attrs.size(0) != positions.size(0):
            raise ValueError(
                "`node_attrs` and `positions` must contain the same number "
                "of atoms."
            )

        if node_attrs.size(1) != self.atomic_numbers.numel():
            raise ValueError(
                "The width of `node_attrs` does not match the DeePMD type "
                "map. Align the dataset using "
                "`AtomisticFeaturizer.align_dataset()` first."
            )

        if node_attrs.numel() > 0:
            binary_entries = torch.all((node_attrs == 0) | (node_attrs == 1))
            if not bool(binary_entries.item()):
                raise ValueError(
                    "`node_attrs` must contain one-hot atomic-type encodings "
                    "with entries equal to zero or one."
                )

            row_sums = node_attrs.sum(dim=-1)
            if not torch.equal(row_sums, torch.ones_like(row_sums)):
                raise ValueError(
                    "Each row of `node_attrs` must contain exactly one active "
                    "atomic type."
                )

        ptr = self._get_ptr(
            data=data,
            n_atoms=positions.size(0),
            device=positions.device,
        )

        if ptr.dim() != 1:
            raise ValueError("Graph `ptr` must be a rank-1 tensor.")

        if (
            ptr.numel() < 2
            or int(ptr[0].item()) != 0
            or int(ptr[-1].item()) != positions.size(0)
        ):
            raise ValueError("Invalid graph `ptr` or `batch` information.")

        if torch.any(ptr[1:] < ptr[:-1]):
            raise ValueError(
                "Graph boundaries in `ptr` must be non-decreasing."
            )

        n_systems = ptr.numel() - 1

        if "batch" in data:
            batch = data["batch"]

            if batch.dim() != 1:
                raise ValueError("`batch` must be a rank-1 tensor.")

            if (
                batch.dtype == torch.bool
                or batch.is_floating_point()
                or batch.is_complex()
            ):
                raise TypeError("`batch` must use an integer dtype.")

            batch = batch.to(device=positions.device, dtype=torch.long)

            if batch.numel() != positions.size(0):
                raise ValueError(
                    "`batch` must contain one system index per atom."
                )

            if batch.numel() > 0:
                if int(batch[0].item()) != 0:
                    raise ValueError(
                        "`batch` system indices must start from zero."
                    )

                if torch.any(batch[1:] < batch[:-1]):
                    raise ValueError(
                        "DeepMDBackbone requires atoms to be grouped by "
                        "system in the batched graph."
                    )

                unique_batch = torch.unique_consecutive(batch)
                expected_batch = torch.arange(
                    unique_batch.numel(),
                    device=batch.device,
                    dtype=batch.dtype,
                )

                if not torch.equal(unique_batch, expected_batch):
                    raise ValueError(
                        "`batch` system indices must be consecutive."
                    )

                if unique_batch.numel() != n_systems:
                    raise ValueError(
                        "`batch` and `ptr` describe different numbers of "
                        "systems."
                    )

                observed_counts = torch.bincount(batch, minlength=n_systems)
                expected_counts = (ptr[1:] - ptr[:-1]).to(
                    device=batch.device,
                    dtype=torch.long,
                )

                if not torch.equal(observed_counts, expected_counts):
                    raise ValueError(
                        "`batch` and `ptr` contain inconsistent atom "
                        "assignments."
                    )

        cells = self._prepare_cells(
            data=data,
            cell=cell,
            n_systems=n_systems,
            positions=positions,
        )
        pbc = self._prepare_pbc(
            data=data,
            cells=cells,
            n_systems=n_systems,
        )

        fully_periodic = torch.all(pbc, dim=1)
        fully_non_periodic = ~torch.any(pbc, dim=1)

        if not bool(torch.all(fully_periodic | fully_non_periodic).item()):
            raise ValueError(
                "DeepMDBackbone currently supports only fully periodic or "
                "fully non-periodic systems. Partial periodic boundary "
                "conditions are unsupported."
            )
