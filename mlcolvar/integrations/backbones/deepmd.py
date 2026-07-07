from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
from torch import nn

from mlcolvar.integrations.atomistic import BaseAtomisticBackbone

from ._utils import (
    to_bool,
    to_float,
    to_int,
    to_int_list,
)


try:
    from ase.data import atomic_numbers as ase_atomic_numbers

    from deepmd.pt.utils import env as deepmd_env
    from deepmd.pt.utils.nlist import (
        extend_input_and_build_neighbor_list,
    )

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


def _type_map_to_atomic_numbers(
    type_map: List[str],
) -> List[int]:
    """Convert a DeePMD element type map to atomic numbers."""

    atomic_number_list: List[int] = []

    for symbol in type_map:
        if symbol not in ase_atomic_numbers:
            raise ValueError(
                "DeepMDBackbone requires every entry in the DeePMD "
                "type map to be a valid chemical-element symbol. "
                f"Could not interpret `{symbol}`."
            )

        atomic_number_list.append(
            int(
                ase_atomic_numbers[
                    symbol
                ]
            )
        )

    if len(
        set(
            atomic_number_list
        )
    ) != len(
        atomic_number_list
    ):
        raise ValueError(
            "The DeePMD type map contains duplicate elements."
        )

    return atomic_number_list


def _is_native_deepmd_model(
    model: nn.Module,
) -> bool:
    """Check for the native DeePMD PyTorch descriptor interface."""

    required_methods = (
        "get_descriptor",
        "get_type_map",
    )

    if not all(
        callable(
            getattr(
                model,
                name,
                None,
            )
        )
        for name in required_methods
    ):
        return False

    try:
        descriptor = model.get_descriptor()

    except Exception:
        return False

    descriptor_methods = (
        "get_rcut",
        "get_sel",
        "get_dim_out",
        "mixed_types",
    )

    return all(
        callable(
            getattr(
                descriptor,
                name,
                None,
            )
        )
        for name in descriptor_methods
    )


def _unwrap_deepmd_model(
    model: nn.Module,
) -> nn.Module:
    """Extract a native DeePMD PyTorch model from common wrappers."""

    if _is_native_deepmd_model(
        model
    ):
        return model

    for attribute in (
        "module",
        "model",
    ):
        candidate = getattr(
            model,
            attribute,
            None,
        )

        if (
            isinstance(
                candidate,
                nn.Module,
            )
            and _is_native_deepmd_model(
                candidate
            )
        ):
            return candidate

    raise ValueError(
        "Could not find a native DeePMD-kit PyTorch model. "
        "The supplied module must expose `get_descriptor()` and "
        "`get_type_map()`. TensorFlow frozen models and the "
        "NumPy-based DeepPotential interface are not supported."
    )


def _infer_descriptor_precision(
    descriptor: nn.Module,
) -> Tuple[str, torch.dtype]:
    """Infer the precision declared by a DeePMD descriptor."""

    descriptor_dtype = getattr(
        descriptor,
        "prec",
        None,
    )

    descriptor_precision = getattr(
        descriptor,
        "precision",
        None,
    )

    if isinstance(
        descriptor_dtype,
        torch.dtype,
    ):
        if descriptor_precision is None:
            descriptor_precision = str(
                descriptor_dtype
            ).replace(
                "torch.",
                "",
            )

        return (
            str(
                descriptor_precision
            ),
            descriptor_dtype,
        )

    if descriptor_precision is not None:
        normalized_precision = str(
            descriptor_precision
        ).lower()

        if normalized_precision in _PRECISION_TO_DTYPE:
            return (
                normalized_precision,
                _PRECISION_TO_DTYPE[
                    normalized_precision
                ],
            )

    for parameter in descriptor.parameters():
        if parameter.is_floating_point():
            return (
                str(
                    parameter.dtype
                ).replace(
                    "torch.",
                    "",
                ),
                parameter.dtype,
            )

    for buffer in descriptor.buffers():
        if buffer.is_floating_point():
            return (
                str(
                    buffer.dtype
                ).replace(
                    "torch.",
                    "",
                ),
                buffer.dtype,
            )

    raise ValueError(
        "Could not infer the floating-point precision of the "
        "DeePMD descriptor."
    )


def _extract_descriptor_features(
    descriptor_output,
) -> torch.Tensor:
    """Extract the atom-level descriptor tensor from DeePMD output."""

    if isinstance(
        descriptor_output,
        tuple,
    ):
        if len(
            descriptor_output
        ) == 0:
            raise RuntimeError(
                "The DeePMD descriptor returned an empty tuple."
            )

        features = descriptor_output[0]

    else:
        features = descriptor_output

    if not isinstance(
        features,
        torch.Tensor,
    ):
        raise RuntimeError(
            "The first DeePMD descriptor output must be a tensor."
        )

    return features


class DeepMDBackbone(
    BaseAtomisticBackbone
):
    """Extract atom-level descriptors from a DeePMD PyTorch model.

    This adapter targets native PyTorch DeePMD models, including DPA-2.
    DeePMD constructs its own padded neighbor list from coordinates,
    atom types, and the simulation cell; the mlcolvar ``edge_index`` is
    therefore not used by this backbone.

    DeePMD's descriptor and neighbor-list implementation can use different
    floating-point precisions:

    - descriptor calculations follow ``descriptor.prec``;
    - periodic ghost-atom construction follows
      ``deepmd_env.GLOBAL_PT_FLOAT_PRECISION``;
    - returned features follow the dtype of the input graph positions.

    The adapter preserves these dtype boundaries while retaining the
    autograd path from output features to the input coordinates.

    Parameters
    ----------
    model
        Native DeePMD-kit PyTorch model exposing ``get_descriptor()``
        and ``get_type_map()``.
    buffer
        Retained for compatibility with the common atomistic-backbone
        interface. DeePMD constructs its own neighbor list.
    long_range_cutoff
        Long-range mlcolvar edges are unsupported because DeePMD builds
        its own neighbor list. This value must remain negative.
    """

    __constants__ = [
        "descriptor_cutoff",
        "descriptor_dim",
        "descriptor_precision",
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
                "DeepMDBackbone requires DeePMD-kit with the "
                "PyTorch backend. Install a DeePMD-kit build that "
                "matches the installed PyTorch version."
            ) from _DEEPMD_IMPORT_ERROR

        if not isinstance(
            model,
            nn.Module,
        ):
            raise TypeError(
                "`model` must be a torch.nn.Module."
            )

        if long_range_cutoff >= 0.0:
            raise ValueError(
                "DeepMDBackbone does not support "
                "`long_range_cutoff` because DeePMD constructs "
                "its own neighbor list."
            )

        model = _unwrap_deepmd_model(
            model
        )

        descriptor = model.get_descriptor()

        raw_type_map = model.get_type_map()

        type_map: List[str] = (
            []
            if raw_type_map is None
            else list(
                raw_type_map
            )
        )

        if len(
            type_map
        ) == 0:
            descriptor_type_map = getattr(
                descriptor,
                "get_type_map",
                None,
            )

            if callable(
                descriptor_type_map
            ):
                raw_descriptor_type_map = (
                    descriptor_type_map()
                )

                if raw_descriptor_type_map is not None:
                    type_map = list(
                        raw_descriptor_type_map
                    )

        if len(
            type_map
        ) == 0:
            raise ValueError(
                "The DeePMD model does not provide a non-empty "
                "element type map."
            )

        atomic_numbers = (
            _type_map_to_atomic_numbers(
                type_map
            )
        )

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

        (
            descriptor_precision,
            descriptor_dtype,
        ) = _infer_descriptor_precision(
            descriptor
        )

        if descriptor_cutoff <= 0.0:
            raise ValueError(
                "The DeePMD descriptor cutoff must be positive, "
                f"found {descriptor_cutoff}."
            )

        if descriptor_dim <= 0:
            raise ValueError(
                "The DeePMD descriptor output dimension must be "
                f"positive, found {descriptor_dim}."
            )

        if (
            not mixed_types
            and len(
                selection
            )
            != len(
                atomic_numbers
            )
        ):
            raise ValueError(
                "For a type-distinguished DeePMD neighbor list, "
                "`descriptor.get_sel()` must contain one entry per "
                "element type. Found "
                f"{len(selection)} selection entries and "
                f"{len(atomic_numbers)} element types."
            )

        super().__init__(
            out_features=descriptor_dim,
            atomic_numbers=atomic_numbers,
            cutoff=descriptor_cutoff,
            sample_kind="atom",
            buffer=buffer,
            long_range_cutoff=-1.0,

            # DeePMD ignores the mlcolvar edge list and reconstructs
            # its own padded neighbor list.
            full_neighbor_list=False,
        )

        self.descriptor_cutoff = (
            descriptor_cutoff
        )

        self.descriptor_dim = (
            descriptor_dim
        )

        self.descriptor_precision = (
            descriptor_precision
        )

        self.selection = selection
        self.mixed_types = mixed_types
        self.type_map = type_map

        self.register_buffer(
            "_descriptor_dtype_reference",
            torch.empty(
                0,
                dtype=descriptor_dtype,
            ),
            persistent=False,
        )

        neighbor_list_dtype = getattr(
            deepmd_env,
            "GLOBAL_PT_FLOAT_PRECISION",
            descriptor_dtype,
        )

        self.register_buffer(
            "_neighbor_list_dtype_reference",
            torch.empty(
                0,
                dtype=neighbor_list_dtype,
            ),
            persistent=False,
        )

        # Register the complete DeePMD model only after nn.Module has
        # been initialized by BaseAtomisticBackbone.
        self.model = model

        # The supplied model may already have been converted to another
        # dtype. Restore the precision declared by the descriptor.
        self._restore_descriptor_precision()

    def _descriptor_dtype(
        self,
    ) -> torch.dtype:
        """Return the internal compute dtype of the descriptor."""

        return (
            self
            ._descriptor_dtype_reference
            .dtype
        )

    def _restore_descriptor_precision(
        self,
    ) -> None:
        """Restore descriptor parameters and buffers to their declared dtype."""

        if not hasattr(
            self,
            "model",
        ):
            return

        descriptor = (
            self.model
            .get_descriptor()
        )

        descriptor_dtype = (
            self._descriptor_dtype()
        )

        descriptor.to(
            dtype=descriptor_dtype
        )

        descriptor_device = None

        for parameter in descriptor.parameters():
            descriptor_device = (
                parameter.device
            )
            break

        if descriptor_device is None:
            for buffer in descriptor.buffers():
                descriptor_device = (
                    buffer.device
                )
                break

        if descriptor_device is not None:
            self._descriptor_dtype_reference = (
                self
                ._descriptor_dtype_reference
                .to(
                    device=descriptor_device,
                    dtype=descriptor_dtype,
                )
            )

    def _apply(
        self,
        fn,
    ):
        """Apply transforms while preserving DeePMD internal precision."""

        module = super()._apply(
            fn
        )

        if hasattr(
            self,
            "model",
        ):
            descriptor_dtype = (
                _PRECISION_TO_DTYPE.get(
                    str(
                        self.descriptor_precision
                    ).lower(),
                    None,
                )
            )

            if descriptor_dtype is None:
                descriptor = (
                    self.model
                    .get_descriptor()
                )

                descriptor_prec = getattr(
                    descriptor,
                    "prec",
                    None,
                )

                if isinstance(
                    descriptor_prec,
                    torch.dtype,
                ):
                    descriptor_dtype = (
                        descriptor_prec
                    )

                else:
                    descriptor_dtype = (
                        self
                        ._descriptor_dtype_reference
                        .dtype
                    )

            self._descriptor_dtype_reference = (
                self
                ._descriptor_dtype_reference
                .to(
                    dtype=descriptor_dtype,
                )
            )

            self._restore_descriptor_precision()

            neighbor_list_dtype = getattr(
                deepmd_env,
                "GLOBAL_PT_FLOAT_PRECISION",
                descriptor_dtype,
            )

            self._neighbor_list_dtype_reference = (
                self
                ._neighbor_list_dtype_reference
                .to(
                    dtype=neighbor_list_dtype,
                )
            )

        return module

    def forward(
        self,
        data: Dict[str, torch.Tensor],
        cell: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute atom-level DeePMD descriptors."""

        if (
            not torch.jit.is_scripting()
            and not torch.jit.is_tracing()
        ):
            self._validate_graph_input(
                data=data,
                cell=cell,
            )

        input_positions = data[
            "positions"
        ]

        output_dtype = (
            input_positions.dtype
        )

        descriptor_dtype = (
            self._descriptor_dtype()
        )

        # This cast preserves the autograd connection to input_positions.
        positions = input_positions.to(
            dtype=descriptor_dtype
        )

        node_attrs = data[
            "node_attrs"
        ]

        ptr = self._get_ptr(
            data=data,
            n_atoms=positions.size(0),
            device=positions.device,
        )

        n_systems = (
            ptr.numel()
            - 1
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

        # align_dataset() arranges node_attrs in the same order as the
        # DeePMD type map, so the one-hot index is the DeePMD type index.
        atom_types = node_attrs.argmax(
            dim=-1
        ).to(
            device=positions.device,
            dtype=torch.long,
        )

        descriptor = (
            self.model
            .get_descriptor()
        )

        feature_blocks: List[
            torch.Tensor
        ] = []

        for system_index in range(
            n_systems
        ):
            start = int(
                ptr[
                    system_index
                ].item()
            )

            end = int(
                ptr[
                    system_index + 1
                ].item()
            )

            system_positions = positions[
                start:end
            ].unsqueeze(
                0
            )

            system_types = atom_types[
                start:end
            ].unsqueeze(
                0
            )

            system_pbc = pbc[
                system_index
            ]

            if bool(
                torch.all(
                    system_pbc
                ).item()
            ):
                box = cells[
                    system_index
                ].unsqueeze(
                    0
                )

            elif bool(
                torch.any(
                    system_pbc
                ).item()
            ):
                raise ValueError(
                    "DeepMDBackbone currently supports only fully "
                    "periodic or fully non-periodic systems. Partial "
                    "periodic boundary conditions are unsupported."
                )

            else:
                box = None

            neighbor_list_dtype = (
                self
                ._neighbor_list_dtype_reference
                .dtype
            )

            # DeePMD creates periodic translation vectors using
            # GLOBAL_PT_FLOAT_PRECISION. Coordinates and box must use the
            # same dtype while the padded neighbor list is constructed.
            neighbor_positions = (
                system_positions.to(
                    dtype=neighbor_list_dtype,
                )
            )

            if box is None:
                neighbor_box = None

            else:
                neighbor_box = box.to(
                    dtype=neighbor_list_dtype,
                )

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

            # Descriptor calculations follow descriptor.prec, which can
            # differ from DeePMD's global neighbor-list precision.
            extended_coord = extended_coord.to(
                dtype=descriptor_dtype,
            )

            descriptor_output = descriptor(
                extended_coord,
                extended_atype,
                neighbor_list,
                mapping=mapping,
            )

            features = (
                _extract_descriptor_features(
                    descriptor_output
                )
            )

            if features.dim() != 3:
                raise RuntimeError(
                    "Expected DeePMD descriptor output with shape "
                    "[n_frames, n_atoms, n_features], but found "
                    f"{tuple(features.shape)}."
                )

            if features.size(0) != 1:
                raise RuntimeError(
                    "DeepMDBackbone processes one system at a time "
                    "and expected one descriptor frame."
                )

            features = features[
                0
            ]

            if features.size(0) != (
                end - start
            ):
                raise RuntimeError(
                    "The number of DeePMD descriptor rows does not "
                    "match the number of local atoms."
                )

            # Return features using the dtype of the input mlcolvar graph.
            feature_blocks.append(
                features.to(
                    dtype=output_dtype,
                )
            )

        if len(
            feature_blocks
        ) == 0:
            return input_positions.new_empty(
                (
                    0,
                    self.out_features,
                )
            )

        output = torch.cat(
            feature_blocks,
            dim=0,
        )

        if (
            not torch.jit.is_scripting()
            and not torch.jit.is_tracing()
        ):
            if output.dim() != 2:
                raise RuntimeError(
                    "Expected the combined DeePMD output to have "
                    "shape [n_atoms, n_features]."
                )

            if output.size(0) != positions.size(0):
                raise RuntimeError(
                    "The total number of DeePMD descriptor rows does "
                    "not match the graph atom count."
                )

            if output.size(1) != self.out_features:
                raise RuntimeError(
                    "Unexpected DeePMD descriptor dimension. "
                    f"Expected {self.out_features}, found "
                    f"{output.size(1)}."
                )

            if output.dtype != output_dtype:
                raise RuntimeError(
                    "DeepMDBackbone failed to restore the graph "
                    f"dtype. Expected {output_dtype}, found "
                    f"{output.dtype}."
                )

        return output

    @torch.jit.unused
    def _validate_graph_input(
        self,
        data: Dict[str, torch.Tensor],
        cell: Optional[torch.Tensor],
    ) -> None:
        """Validate graph fields required by the DeePMD adapter."""

        required_keys = [
            "positions",
            "node_attrs",
        ]

        missing_keys = [
            key
            for key in required_keys
            if key not in data
        ]

        if missing_keys:
            raise KeyError(
                "DeepMD graph input is missing required fields: "
                f"{missing_keys}."
            )

        positions = data[
            "positions"
        ]

        node_attrs = data[
            "node_attrs"
        ]

        if (
            positions.dim() != 2
            or positions.size(1) != 3
        ):
            raise ValueError(
                "`positions` must have shape [n_atoms, 3], "
                f"found {tuple(positions.shape)}."
            )

        if not positions.is_floating_point():
            raise ValueError(
                "`positions` must be a floating-point tensor."
            )

        if node_attrs.dim() != 2:
            raise ValueError(
                "`node_attrs` must be a rank-2 tensor."
            )

        if (
            node_attrs.size(0)
            != positions.size(0)
        ):
            raise ValueError(
                "`node_attrs` and `positions` must contain the "
                "same number of atoms."
            )

        if (
            node_attrs.size(1)
            != self.atomic_numbers.numel()
        ):
            raise ValueError(
                "The width of `node_attrs` does not match the "
                "DeePMD type map. Align the dataset using "
                "`AtomisticFeaturizer.align_dataset()` first."
            )

        row_sums = node_attrs.sum(
            dim=-1
        )

        if not torch.allclose(
            row_sums,
            torch.ones_like(
                row_sums
            ),
        ):
            raise ValueError(
                "`node_attrs` must contain one-hot atomic-type "
                "encodings."
            )

        max_values = node_attrs.max(
            dim=-1
        ).values

        if not torch.allclose(
            max_values,
            torch.ones_like(
                max_values
            ),
        ):
            raise ValueError(
                "Every `node_attrs` row must contain exactly one "
                "active atomic type."
            )

        ptr = self._get_ptr(
            data=data,
            n_atoms=positions.size(0),
            device=positions.device,
        )

        if (
            ptr.numel() < 2
            or int(
                ptr[0].item()
            ) != 0
            or int(
                ptr[-1].item()
            )
            != positions.size(0)
        ):
            raise ValueError(
                "Invalid graph `ptr` or `batch` information."
            )

        if torch.any(
            ptr[1:]
            < ptr[:-1]
        ):
            raise ValueError(
                "Graph boundaries in `ptr` must be non-decreasing."
            )

        if "batch" in data:
            batch = data[
                "batch"
            ].to(
                dtype=torch.long
            )

            if (
                batch.numel()
                != positions.size(0)
            ):
                raise ValueError(
                    "`batch` must contain one system index per atom."
                )

            if (
                batch.numel() > 1
                and torch.any(
                    batch[1:]
                    < batch[:-1]
                )
            ):
                raise ValueError(
                    "DeepMDBackbone requires atoms to be grouped by "
                    "system in the batched graph."
                )

        self._prepare_cells(
            data=data,
            cell=cell,
            n_systems=ptr.numel() - 1,
            positions=positions,
        )
