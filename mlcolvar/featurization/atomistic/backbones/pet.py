from __future__ import annotations

# Apply the PET/TorchScript compatibility patch before importing metatensor or metatomic.
from ..patches import pet_jit as _pet_jit  # noqa: F401

from typing import Dict, List, Optional, Tuple

import torch
from torch import nn

from .base import BaseAtomisticBackbone
from ..graph import (
    get_graph_ptr,
    prepare_cells,
    prepare_pbc,
)

from .utils import (
    to_bool,
    to_float,
    to_int,
    to_int_list,
)


try:
    from metatensor.torch import Labels, TensorBlock
    from metatomic.torch import ModelOutput, System

    _METATOMIC_AVAILABLE = True
    _METATOMIC_IMPORT_ERROR = None

except ImportError as exc:
    Labels = None
    TensorBlock = None
    ModelOutput = None
    System = None

    _METATOMIC_AVAILABLE = False
    _METATOMIC_IMPORT_ERROR = exc


__all__ = ["PETBackbone"]


_PRECISION_TO_DTYPE = {
    "float16": torch.float16,
    "float32": torch.float32,
    "float64": torch.float64,
    "bfloat16": torch.bfloat16,
}

_DTYPE_TO_PRECISION = {
    dtype: precision
    for precision, dtype in _PRECISION_TO_DTYPE.items()
}


def _single_floating_dtype(
    tensors,
    name: str,
) -> Optional[torch.dtype]:
    """Return the unique floating dtype in an iterable of tensors."""

    dtypes = {
        tensor.dtype
        for tensor in tensors
        if tensor.is_floating_point()
    }

    if len(dtypes) > 1:
        raise ValueError(
            f"The PET model contains {name} with mixed floating-point "
            f"precisions: {sorted(str(dtype) for dtype in dtypes)}."
        )

    if len(dtypes) == 0:
        return None

    return next(iter(dtypes))


def _infer_model_precision(
    model: nn.Module,
) -> Tuple[str, torch.dtype]:
    """Infer PET's internal floating-point precision."""

    model_dtype = _single_floating_dtype(
        model.parameters(),
        name="parameters",
    )

    if model_dtype is None:
        model_dtype = _single_floating_dtype(
            model.buffers(),
            name="buffers",
        )

    if model_dtype is None:
        raise ValueError(
            "Could not infer the floating-point precision of the PET model."
        )

    if model_dtype not in _DTYPE_TO_PRECISION:
        raise ValueError(
            "Unsupported PET floating-point dtype: "
            f"{model_dtype}."
        )

    return _DTYPE_TO_PRECISION[model_dtype], model_dtype


def _is_native_pet_model(
    model: nn.Module,
) -> bool:
    """Check whether a module exposes the native PET public interface."""

    required_attributes = (
        "atomic_types",
        "cutoff",
        "d_node",
        "d_pet",
        "num_readout_layers",
        "supported_outputs",
        "requested_neighbor_lists",
    )

    if not all(hasattr(model, name) for name in required_attributes):
        return False

    try:
        return "feature" in model.supported_outputs()
    except Exception:
        return False


def _unwrap_pet_model(
    model: nn.Module,
) -> nn.Module:
    """Extract a native PET model from an optional wrapper."""

    if _is_native_pet_model(model):
        return model

    wrapped_model = getattr(model, "model", None)

    if isinstance(wrapped_model, nn.Module) and _is_native_pet_model(
        wrapped_model
    ):
        return wrapped_model

    raise ValueError(
        "Could not find a native metatrain PET model. The supplied module "
        "must expose `atomic_types`, `cutoff`, `d_node`, `d_pet`, "
        "`num_readout_layers`, `supported_outputs`, and "
        "`requested_neighbor_lists`."
    )


def _infer_pet_layout(
    model: nn.Module,
) -> Tuple[int, int, int]:
    """Infer PET public feature dimensions."""

    d_node = to_int(
        model.d_node,
        name="model.d_node",
    )

    d_pet = to_int(
        model.d_pet,
        name="model.d_pet",
    )

    num_readout_layers = to_int(
        model.num_readout_layers,
        name="model.num_readout_layers",
    )

    if d_node <= 0:
        raise ValueError(
            "PET `d_node` must be positive, "
            f"found {d_node}."
        )

    if d_pet <= 0:
        raise ValueError(
            "PET `d_pet` must be positive, "
            f"found {d_pet}."
        )

    if num_readout_layers <= 0:
        raise ValueError(
            "PET `num_readout_layers` must be positive, "
            f"found {num_readout_layers}."
        )

    return d_node, d_pet, num_readout_layers


class PETBackbone(BaseAtomisticBackbone):
    """Extract atom-level features from a pretrained metatrain PET model."""

    __constants__ = [
        "d_node",
        "d_pet",
        "num_readout_layers",
        "neighbor_cutoff",
        "neighbor_full_list",
        "neighbor_strict",
        "model_precision",
    ]

    def __init__(
        self,
        model: nn.Module,
        buffer: float = 0.0,
        long_range_cutoff: float = -1.0,
    ) -> None:
        if not _METATOMIC_AVAILABLE:
            raise ImportError(
                "PETBackbone requires the optional PET dependencies. "
                'Install them with `pip install "metatrain[pet]"`.'
            ) from _METATOMIC_IMPORT_ERROR

        if not isinstance(model, nn.Module):
            raise TypeError("`model` must be a torch.nn.Module.")

        model = _unwrap_pet_model(model)

        d_node, d_pet, num_readout_layers = _infer_pet_layout(model)

        atomic_numbers = to_int_list(
            model.atomic_types,
            name="model.atomic_types",
        )

        requested_neighbor_lists = model.requested_neighbor_lists()

        if len(requested_neighbor_lists) != 1:
            raise ValueError(
                "PETBackbone expects PET to request exactly one neighbor "
                f"list, but found {len(requested_neighbor_lists)}."
            )

        neighbor_options = requested_neighbor_lists[0]

        neighbor_cutoff = to_float(
            neighbor_options.cutoff,
            name="PET neighbor-list cutoff",
        )

        neighbor_full_list = to_bool(
            neighbor_options.full_list,
            name="PET neighbor-list full_list",
        )

        neighbor_strict = to_bool(
            neighbor_options.strict,
            name="PET neighbor-list strict",
        )

        model_cutoff = to_float(
            model.cutoff,
            name="model.cutoff",
        )

        if abs(model_cutoff - neighbor_cutoff) > 1e-12:
            raise ValueError(
                "PET model cutoff and requested neighbor-list cutoff "
                "do not match: "
                f"model.cutoff={model_cutoff}, "
                f"neighbor cutoff={neighbor_cutoff}."
            )

        model_precision, model_dtype = _infer_model_precision(model)

        out_features = num_readout_layers * (d_node + d_pet)

        super().__init__(
            out_features=out_features,
            atomic_numbers=atomic_numbers,
            cutoff=neighbor_cutoff,
            sample_kind="atom",
            buffer=buffer,
            long_range_cutoff=long_range_cutoff,
            full_neighbor_list=neighbor_full_list,
        )

        self.d_node = d_node
        self.d_pet = d_pet
        self.num_readout_layers = num_readout_layers

        self.neighbor_cutoff = neighbor_cutoff
        self.neighbor_full_list = neighbor_full_list
        self.neighbor_strict = neighbor_strict
        self.neighbor_options = neighbor_options
        self.model_precision = model_precision

        self.register_buffer(
            "_model_dtype_reference",
            torch.empty(
                0,
                dtype=model_dtype,
            ),
            persistent=False,
        )

        self.model = model
        self._restore_model_precision()

    @torch.jit.unused
    def _model_dtype(self) -> torch.dtype:
        """Return PET's fixed internal dtype."""

        return _PRECISION_TO_DTYPE[self.model_precision]

    @torch.jit.unused
    def _restore_model_precision(self) -> None:
        """Restore PET to its construction-time precision."""

        if not hasattr(self, "model"):
            return

        model_dtype = self._model_dtype()
        self.model.to(dtype=model_dtype)

        model_device = None

        for parameter in self.model.parameters():
            model_device = parameter.device
            break

        if model_device is None:
            for buffer in self.model.buffers():
                model_device = buffer.device
                break

        if model_device is not None:
            self._model_dtype_reference = self._model_dtype_reference.to(
                device=model_device,
                dtype=model_dtype,
            )

    def _apply(
        self,
        fn,
        recurse: bool = True,
    ):
        """Apply transforms while preserving PET's internal dtype."""

        module = super()._apply(
            fn,
            recurse=recurse,
        )

        if hasattr(self, "model"):
            self._restore_model_precision()

        return module

    def forward(
        self,
        data: Dict[str, torch.Tensor],
        cell: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute PET atom-level features."""

        input_positions = data["positions"]

        output_dtype = input_positions.dtype
        output_device = input_positions.device

        model_dtype = self._model_dtype_reference.dtype
        model_device = self._model_dtype_reference.device

        positions = input_positions.to(
            device=model_device,
            dtype=model_dtype,
        )

        ptr = get_graph_ptr(
            data=data,
            n_atoms=positions.size(0),
            device=model_device,
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

        atomic_numbers = self.atomic_numbers.to(
            device=model_device,
        )

        species_indices = data["node_attrs"].argmax(dim=-1).to(
            device=model_device,
            dtype=torch.long,
        )

        atom_types = atomic_numbers[species_indices].to(dtype=torch.int32)

        edge_index = data["edge_index"].to(
            device=model_device,
            dtype=torch.long,
        )

        unit_shifts = data["unit_shifts"].to(
            device=model_device,
            dtype=model_dtype,
        )

        long_range_mask = torch.jit.annotate(
            Optional[torch.Tensor],
            None,
        )

        if "edge_masks_lr" in data:
            long_range_mask = data["edge_masks_lr"].reshape(-1).to(
                device=model_device,
                dtype=torch.bool,
            )

        systems = torch.jit.annotate(
            List[System],
            [],
        )

        for system_index in range(n_systems):
            start = int(ptr[system_index].item())
            end = int(ptr[system_index + 1].item())

            system_positions = positions[start:end]
            system_cell = cells[system_index]

            system = System(
                types=atom_types[start:end],
                positions=system_positions,
                cell=system_cell,
                pbc=pbc[system_index],
            )

            neighbors = self._build_neighbor_list(
                edge_index=edge_index,
                unit_shifts=unit_shifts,
                long_range_mask=long_range_mask,
                positions=system_positions,
                cell=system_cell,
                start=start,
                end=end,
            )

            system.add_neighbor_list(
                self.neighbor_options,
                neighbors,
            )

            systems.append(system)

        result = self.model(
            systems=systems,
            outputs={
                "feature": ModelOutput(
                    sample_kind="atom",
                ),
            },
            selected_atoms=None,
        )

        if "feature" not in result:
            raise RuntimeError(
                "The PET model did not return the requested `feature` output."
            )

        return result["feature"].block().values.to(
            device=output_device,
            dtype=output_dtype,
        )

    def _build_neighbor_list(
        self,
        edge_index: torch.Tensor,
        unit_shifts: torch.Tensor,
        long_range_mask: Optional[torch.Tensor],
        positions: torch.Tensor,
        cell: torch.Tensor,
        start: int,
        end: int,
    ) -> TensorBlock:
        """Convert one mlcolvar graph into a metatomic neighbor list."""

        first_global = edge_index[0]
        second_global = edge_index[1]

        edge_mask = (
            (first_global >= start)
            & (first_global < end)
            & (second_global >= start)
            & (second_global < end)
        )

        if long_range_mask is not None:
            edge_mask = edge_mask & ~long_range_mask

        first_atom = first_global[edge_mask] - start
        second_atom = second_global[edge_mask] - start

        cell_shifts = torch.round(unit_shifts[edge_mask]).to(
            dtype=torch.int32,
        )

        edge_vectors = (
            positions[second_atom]
            - positions[first_atom]
            + cell_shifts.to(dtype=positions.dtype) @ cell
        )

        if self.neighbor_strict:
            cutoff_mask = (
                torch.linalg.vector_norm(
                    edge_vectors,
                    dim=-1,
                )
                <= self.neighbor_cutoff
            )

            first_atom = first_atom[cutoff_mask]
            second_atom = second_atom[cutoff_mask]
            cell_shifts = cell_shifts[cutoff_mask]
            edge_vectors = edge_vectors[cutoff_mask]

        samples = Labels(
            names=[
                "first_atom",
                "second_atom",
                "cell_shift_a",
                "cell_shift_b",
                "cell_shift_c",
            ],
            values=torch.cat(
                (
                    first_atom.to(dtype=torch.int32).reshape(-1, 1),
                    second_atom.to(dtype=torch.int32).reshape(-1, 1),
                    cell_shifts,
                ),
                dim=1,
            ),
        )

        components = torch.jit.annotate(
            List[Labels],
            [
                Labels(
                    names=["xyz"],
                    values=torch.arange(
                        3,
                        device=positions.device,
                        dtype=torch.int32,
                    ).reshape(-1, 1),
                )
            ],
        )

        properties = Labels(
            names=["distance"],
            values=torch.zeros(
                (1, 1),
                device=positions.device,
                dtype=torch.int32,
            ),
        )

        return TensorBlock(
            values=edge_vectors.reshape(-1, 3, 1),
            samples=samples,
            components=components,
            properties=properties,
        )
