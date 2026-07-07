"""PET integration for pretrained atomistic representations."""

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
    from metatensor.torch import Labels, TensorBlock
    from metatomic.torch import ModelOutput, NeighborListOptions, System

    _METATOMIC_AVAILABLE = True
    _METATOMIC_IMPORT_ERROR = None

except ImportError as exc:
    # Keep mlcolvar importable when the optional PET dependencies are
    # unavailable.
    Labels = None
    TensorBlock = None
    ModelOutput = None
    NeighborListOptions = None
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
    torch.float16: "float16",
    torch.float32: "float32",
    torch.float64: "float64",
    torch.bfloat16: "bfloat16",
}


def _infer_model_precision(
    model: nn.Module,
) -> Tuple[str, torch.dtype]:
    """Infer the floating-point precision used by a PET model.

    Floating-point parameters are used preferentially. Floating-point
    buffers are considered only when the model has no floating-point
    parameters.

    Parameters
    ----------
    model
        PET model whose internal precision should be inferred.

    Returns
    -------
    precision
        String representation of the inferred precision.
    dtype
        Corresponding PyTorch dtype.

    Raises
    ------
    ValueError
        If no floating-point dtype can be inferred, multiple parameter
        precisions are detected, or the inferred dtype is unsupported.
    """
    parameter_dtypes = {
        parameter.dtype
        for parameter in model.parameters()
        if parameter.is_floating_point()
    }

    if len(parameter_dtypes) > 1:
        raise ValueError(
            "The PET model contains parameters with mixed floating-point "
            "precisions: "
            f"{sorted(str(dtype) for dtype in parameter_dtypes)}."
        )

    if parameter_dtypes:
        model_dtype = next(iter(parameter_dtypes))

    else:
        buffer_dtypes = {
            buffer.dtype
            for buffer in model.buffers()
            if buffer.is_floating_point()
        }

        if len(buffer_dtypes) == 0:
            raise ValueError(
                "Could not infer the floating-point precision of the "
                "PET model."
            )

        if len(buffer_dtypes) > 1:
            raise ValueError(
                "The PET model contains buffers with mixed floating-point "
                "precisions: "
                f"{sorted(str(dtype) for dtype in buffer_dtypes)}."
            )

        model_dtype = next(iter(buffer_dtypes))

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

    if not all(
        hasattr(model, attribute)
        for attribute in required_attributes
    ):
        return False

    try:
        outputs = model.supported_outputs()

    except Exception:
        # This function probes potentially incompatible wrappers. Errors
        # raised while querying an unsupported interface should not escape.
        return False

    return "feature" in outputs


def _unwrap_pet_model(
    model: nn.Module,
) -> nn.Module:
    """Extract a native PET model from an optional wrapper.

    PET-MAD checkpoints can be wrapped by modules such as
    ``LLPRUncertaintyModel``. In that case, the native PET model is
    available through ``wrapper.model``.

    Parameters
    ----------
    model
        Native PET model or supported wrapper.

    Returns
    -------
    nn.Module
        Native PET model.

    Raises
    ------
    ValueError
        If no native PET model can be found.
    """
    if _is_native_pet_model(model):
        return model

    wrapped_model = getattr(
        model,
        "model",
        None,
    )

    if (
        isinstance(wrapped_model, nn.Module)
        and _is_native_pet_model(wrapped_model)
    ):
        return wrapped_model

    raise ValueError(
        "Could not find a native metatrain PET model. The supplied "
        "module must expose `atomic_types`, `cutoff`, `d_node`, "
        "`d_pet`, `num_readout_layers`, `supported_outputs`, and "
        "`requested_neighbor_lists`. PET-MAD LLPR wrappers are "
        "supported automatically through their `.model` attribute."
    )


def _infer_pet_layout(
    model: nn.Module,
) -> Tuple[int, int, int]:
    """Infer and validate the dimension of PET's public feature output.

    Parameters
    ----------
    model
        Native PET model.

    Returns
    -------
    d_node
        Dimension of each node-feature block.
    d_pet
        Dimension of each edge-derived PET feature block.
    num_readout_layers
        Number of readout layers represented in the public feature output.
    """
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
    """Extract atom-level features from a pretrained metatrain PET model.

    This adapter uses PET's public metatomic interface::

        mlcolvar graph
            -> metatomic Systems
            -> PET "feature" output

    The mlcolvar graph neighbor list is converted into the metatomic
    ``TensorBlock`` requested by PET. PET then performs its own input
    preprocessing and message passing.

    PET is kept in the floating-point precision detected when this
    backbone is constructed. Device transfers requested through
    ``to(device=...)`` are retained, while surrounding calls such as
    ``float()`` or ``double()`` do not change PET's internal precision.

    Pooling, masking, parameter freezing, and graph-level output are
    handled by
    :class:`mlcolvar.integrations.atomistic.AtomisticFeaturizer`.

    Parameters
    ----------
    model
        Native ``metatrain.pet.model.PET`` model or a wrapper containing
        the native PET model in ``model.model``. PET-MAD LLPR checkpoints
        are accepted directly.
    buffer
        Additional environment buffer used during mlcolvar graph
        construction.
    long_range_cutoff
        Optional mlcolvar long-range graph cutoff. Long-range edges marked
        by ``edge_masks_lr`` are excluded from the PET neighbor list.
    """

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
            raise TypeError(
                "`model` must be a torch.nn.Module."
            )

        # PET-MAD checkpoints can be wrapped by LLPRUncertaintyModel.
        model = _unwrap_pet_model(model)

        (
            d_node,
            d_pet,
            num_readout_layers,
        ) = _infer_pet_layout(model)

        atomic_numbers = to_int_list(
            model.atomic_types,
            name="model.atomic_types",
        )

        requested_neighbor_lists = (
            model.requested_neighbor_lists()
        )

        if len(requested_neighbor_lists) != 1:
            raise ValueError(
                "PETBackbone currently expects PET to request exactly "
                "one neighbor list, but found "
                f"{len(requested_neighbor_lists)}."
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

        (
            model_precision,
            model_dtype,
        ) = _infer_model_precision(model)

        # PET concatenates one d_node node-feature block and one d_pet
        # cutoff-weighted edge-feature block for every readout layer.
        out_features = (
            num_readout_layers
            * (d_node + d_pet)
        )

        # BaseAtomisticBackbone initializes nn.Module before the PET model
        # is registered as a child module.
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

        # Cache the exact options requested by PET. Querying
        # requested_neighbor_lists() dynamically inside forward would make
        # the adapter depend on this Python-facing method being exported by
        # the wrapped model.
        self.neighbor_options = neighbor_options

        # This is the precision observed when PETBackbone is constructed,
        # not necessarily the original dtype of the checkpoint file.
        self.model_precision = model_precision

        # This non-persistent buffer tracks both PET's current device and
        # its fixed internal floating-point dtype.
        self.register_buffer(
            "_model_dtype_reference",
            torch.empty(
                0,
                dtype=model_dtype,
            ),
            persistent=False,
        )

        self.model = model

        # Ensure that the registered model uses the construction-time
        # internal precision.
        self._restore_model_precision()

    @torch.jit.unused
    def _model_dtype(
        self,
    ) -> torch.dtype:
        """Return PET's fixed internal floating-point dtype."""
        return _PRECISION_TO_DTYPE[
            self.model_precision
        ]

    @torch.jit.unused
    def _restore_model_precision(
        self,
    ) -> None:
        """Restore PET to its construction-time floating-point precision."""
        if not hasattr(self, "model"):
            return

        model_dtype = self._model_dtype()

        # A dtype-only conversion preserves the model's current device.
        self.model.to(
            dtype=model_dtype,
        )

        model_device = None

        for parameter in self.model.parameters():
            model_device = parameter.device
            break

        if model_device is None:
            for buffer in self.model.buffers():
                model_device = buffer.device
                break

        if model_device is not None:
            self._model_dtype_reference = (
                self._model_dtype_reference.to(
                    device=model_device,
                    dtype=model_dtype,
                )
            )

    def _apply(
        self,
        fn,
        recurse: bool = True,
    ):
        """Apply module transforms while preserving PET's internal dtype.

        Calls such as ``to(device=...)``, ``float()``, ``double()``, or
        ``to(dtype=...)`` recursively reach the PET model. Device changes
        are retained, but PET is restored to the precision detected when
        this backbone was constructed.
        """
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
        """Compute PET atom-level features.

        Parameters
        ----------
        data
            Batched mlcolvar graph containing ``positions``,
            ``node_attrs``, ``edge_index``, and ``unit_shifts``.
            ``ptr`` or ``batch`` is used to separate systems.
        cell
            Optional runtime cell. When supplied, it overrides
            ``data["cell"]``.

        Returns
        -------
        torch.Tensor
            PET features with shape ``[n_atoms, out_features]``.
        """
        if (
            not torch.jit.is_scripting()
            and not torch.jit.is_tracing()
        ):
            self._validate_graph_input(
                data=data,
                cell=cell,
            )

        input_positions = data["positions"]

        output_dtype = input_positions.dtype
        output_device = input_positions.device

        # Use the reference buffer directly in the scripted forward path,
        # avoiding a Python dictionary lookup from string to torch.dtype.
        model_dtype = self._model_dtype_reference.dtype
        model_device = self._model_dtype_reference.device

        # PET commonly operates in float32. Casting geometry preserves the
        # autograd connection to the original input coordinates.
        positions = input_positions.to(
            device=model_device,
            dtype=model_dtype,
        )

        ptr = self._get_ptr(
            data=data,
            n_atoms=positions.size(0),
            device=model_device,
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

        node_attrs = data["node_attrs"]

        atomic_numbers = self.atomic_numbers.to(
            device=model_device,
        )

        species_indices = node_attrs.argmax(
            dim=-1,
        ).to(
            device=model_device,
            dtype=torch.long,
        )

        atom_types = atomic_numbers[
            species_indices
        ].to(
            dtype=torch.int32,
        )

        # Move global edge data once rather than once per system.
        edge_index = data["edge_index"].to(
            device=model_device,
            dtype=torch.long,
        )

        unit_shifts = data["unit_shifts"].to(
            device=model_device,
            dtype=model_dtype,
        )

        long_range_mask: Optional[torch.Tensor] = None

        if "edge_masks_lr" in data:
            long_range_mask = (
                data["edge_masks_lr"]
                .reshape(-1)
                .to(
                    device=model_device,
                    dtype=torch.bool,
                )
            )

        systems = torch.jit.annotate(
            List[System],
            [],
        )

        for system_index in range(n_systems):
            start = int(
                ptr[system_index].item()
            )

            end = int(
                ptr[system_index + 1].item()
            )

            system_positions = positions[start:end]
            system_types = atom_types[start:end]
            system_cell = cells[system_index]
            system_pbc = pbc[system_index]

            system = System(
                types=system_types,
                positions=system_positions,
                cell=system_cell,
                pbc=system_pbc,
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

        requested_outputs: Dict[str, ModelOutput] = {
            "feature": ModelOutput(
                sample_kind="atom",
            ),
        }

        result = self.model(
            systems=systems,
            outputs=requested_outputs,
            selected_atoms=None,
        )

        if "feature" not in result:
            raise RuntimeError(
                "The PET model did not return the requested "
                "`feature` output."
            )

        features = (
            result["feature"]
            .block()
            .values
        )

        if (
            not torch.jit.is_scripting()
            and not torch.jit.is_tracing()
        ):
            self._validate_feature_output(
                features=features,
                n_atoms=positions.size(0),
            )

        # Restore the dtype and device expected by the surrounding
        # mlcolvar model. This cast preserves coordinate gradients.
        return features.to(
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
        """Convert one mlcolvar graph into a metatomic neighbor list.

        Parameters
        ----------
        edge_index
            Global graph edge indices with shape ``[2, n_edges]``.
        unit_shifts
            Integer-valued periodic cell offsets with shape
            ``[n_edges, 3]``.
        long_range_mask
            Optional mask marking mlcolvar long-range edges that should not
            be passed to PET.
        positions
            Positions belonging to the current system.
        cell
            Cell matrix of the current system.
        start
            Global index of the first atom in the current system.
        end
            Exclusive global index of the final atom in the current
            system.
        """
        first_global = edge_index[0]
        second_global = edge_index[1]

        edge_mask = (
            (first_global >= start)
            & (first_global < end)
            & (second_global >= start)
            & (second_global < end)
        )

        # PET receives only its requested short-range neighbor list.
        if long_range_mask is not None:
            edge_mask = (
                edge_mask
                & ~long_range_mask
            )

        first_atom = (
            first_global[edge_mask]
            - start
        )

        second_atom = (
            second_global[edge_mask]
            - start
        )

        cell_shifts = torch.round(
            unit_shifts[edge_mask],
        ).to(
            dtype=torch.int32,
        )

        cartesian_shifts = (
            cell_shifts.to(
                dtype=positions.dtype,
            )
            @ cell
        )

        edge_vectors = (
            positions[second_atom]
            - positions[first_atom]
            + cartesian_shifts
        )

        # A strict PET neighbor list excludes graph-buffer edges and edges
        # generated using a larger mlcolvar cutoff.
        if self.neighbor_strict:
            distances = torch.linalg.vector_norm(
                edge_vectors,
                dim=-1,
            )

            cutoff_mask = (
                distances
                <= self.neighbor_cutoff
            )

            first_atom = first_atom[
                cutoff_mask
            ]

            second_atom = second_atom[
                cutoff_mask
            ]

            cell_shifts = cell_shifts[
                cutoff_mask
            ]

            edge_vectors = edge_vectors[
                cutoff_mask
            ]

        sample_values = torch.cat(
            [
                first_atom.to(
                    dtype=torch.int32,
                ).reshape(-1, 1),
                second_atom.to(
                    dtype=torch.int32,
                ).reshape(-1, 1),
                cell_shifts,
            ],
            dim=1,
        )

        samples = Labels(
            names=[
                "first_atom",
                "second_atom",
                "cell_shift_a",
                "cell_shift_b",
                "cell_shift_c",
            ],
            values=sample_values,
        )

        components = torch.jit.annotate(
            List[Labels],
            [],
        )

        components.append(
            Labels(
                names=["xyz"],
                values=torch.arange(
                    3,
                    device=positions.device,
                    dtype=torch.int32,
                ).reshape(-1, 1),
            )
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
            values=edge_vectors.reshape(
                -1,
                3,
                1,
            ),
            samples=samples,
            components=components,
            properties=properties,
        )

    @torch.jit.unused
    def _validate_feature_output(
        self,
        features: torch.Tensor,
        n_atoms: int,
    ) -> None:
        """Validate the public PET feature tensor in eager mode."""
        if not isinstance(features, torch.Tensor):
            raise RuntimeError(
                "PET `feature` values must be a torch.Tensor."
            )

        if features.dim() != 2:
            raise RuntimeError(
                "Expected PET `feature` values to have shape "
                "[n_atoms, n_features], but found "
                f"{tuple(features.shape)}."
            )

        if features.size(0) != n_atoms:
            raise RuntimeError(
                "The number of PET feature rows does not match the "
                "number of graph atoms. Expected "
                f"{n_atoms}, found {features.size(0)}."
            )

        if features.size(1) != self.out_features:
            raise RuntimeError(
                "Unexpected PET feature dimension. Expected "
                f"{self.out_features}, found "
                f"{features.size(1)}."
            )

    @torch.jit.unused
    def _validate_graph_input(
        self,
        data: Dict[str, torch.Tensor],
        cell: Optional[torch.Tensor],
    ) -> None:
        """Validate graph fields required by the PET adapter."""
        required_keys = (
            "positions",
            "node_attrs",
            "edge_index",
            "unit_shifts",
        )

        missing_keys = [
            key
            for key in required_keys
            if key not in data
        ]

        if missing_keys:
            raise KeyError(
                "PET graph input is missing required fields: "
                f"{missing_keys}."
            )

        positions = data["positions"]
        node_attrs = data["node_attrs"]
        edge_index = data["edge_index"]
        unit_shifts = data["unit_shifts"]

        if (
            positions.dim() != 2
            or positions.size(1) != 3
        ):
            raise ValueError(
                "`positions` must have shape [n_atoms, 3], "
                f"found {tuple(positions.shape)}."
            )

        if not positions.is_floating_point():
            raise TypeError(
                "`positions` must use a floating-point dtype."
            )

        if node_attrs.dim() != 2:
            raise ValueError(
                "`node_attrs` must be a rank-2 tensor, "
                f"found shape {tuple(node_attrs.shape)}."
            )

        if node_attrs.size(0) != positions.size(0):
            raise ValueError(
                "`node_attrs` and `positions` must contain the "
                "same number of atoms."
            )

        if node_attrs.size(1) != self.atomic_numbers.numel():
            raise ValueError(
                "The width of `node_attrs` does not match the PET "
                "atomic-type table. Align the dataset with "
                "`AtomisticFeaturizer.align_dataset()` before "
                "constructing the datamodule."
            )

        if node_attrs.numel() > 0:
            binary_entries = torch.all(
                (node_attrs == 0)
                | (node_attrs == 1)
            )

            if not bool(binary_entries.item()):
                raise ValueError(
                    "`node_attrs` must contain one-hot atomic-type "
                    "encodings with entries equal to zero or one."
                )

            row_sums = node_attrs.sum(
                dim=1,
            )

            if not torch.equal(
                row_sums,
                torch.ones_like(row_sums),
            ):
                raise ValueError(
                    "Each row of `node_attrs` must contain exactly "
                    "one active atomic type."
                )

        if (
            edge_index.dim() != 2
            or edge_index.size(0) != 2
        ):
            raise ValueError(
                "`edge_index` must have shape [2, n_edges], "
                f"found {tuple(edge_index.shape)}."
            )

        if (
            edge_index.dtype == torch.bool
            or edge_index.is_floating_point()
            or edge_index.is_complex()
        ):
            raise TypeError(
                "`edge_index` must use an integer dtype."
            )

        if (
            unit_shifts.dim() != 2
            or unit_shifts.size(1) != 3
        ):
            raise ValueError(
                "`unit_shifts` must have shape [n_edges, 3], "
                f"found {tuple(unit_shifts.shape)}."
            )

        if unit_shifts.size(0) != edge_index.size(1):
            raise ValueError(
                "`unit_shifts` and `edge_index` contain different "
                "numbers of edges."
            )

        shifts_for_validation = unit_shifts.to(
            dtype=torch.float64,
        )

        if not torch.allclose(
            shifts_for_validation,
            torch.round(shifts_for_validation),
        ):
            raise ValueError(
                "`unit_shifts` must contain integer-valued periodic "
                "cell offsets."
            )

        if "edge_masks_lr" in data:
            if (
                data["edge_masks_lr"].numel()
                != edge_index.size(1)
            ):
                raise ValueError(
                    "`edge_masks_lr` must contain one entry per edge."
                )

        n_atoms = positions.size(0)

        if edge_index.numel() > 0:
            edge_index_long = edge_index.to(
                dtype=torch.long,
            )

            minimum_index = int(
                edge_index_long.min().item()
            )

            maximum_index = int(
                edge_index_long.max().item()
            )

            if minimum_index < 0:
                raise ValueError(
                    "`edge_index` contains negative atom indices."
                )

            if maximum_index >= n_atoms:
                raise ValueError(
                    "`edge_index` contains atom indices outside the "
                    f"graph. The maximum valid index is {n_atoms - 1}, "
                    f"but found {maximum_index}."
                )

        ptr = self._get_ptr(
            data=data,
            n_atoms=n_atoms,
            device=positions.device,
        )

        if ptr.dim() != 1:
            raise ValueError(
                "Graph `ptr` must be a rank-1 tensor."
            )

        if (
            ptr.numel() < 2
            or int(ptr[0].item()) != 0
            or int(ptr[-1].item()) != n_atoms
        ):
            raise ValueError(
                "Invalid graph `ptr`/`batch` information."
            )

        if torch.any(
            ptr[1:] < ptr[:-1]
        ):
            raise ValueError(
                "Graph `ptr` must be monotonically non-decreasing."
            )

        n_systems = ptr.numel() - 1

        atom_systems: Optional[torch.Tensor] = None

        if "batch" in data:
            batch = data["batch"]

            if batch.dim() != 1:
                raise ValueError(
                    "`batch` must be a rank-1 tensor."
                )

            if (
                batch.dtype == torch.bool
                or batch.is_floating_point()
                or batch.is_complex()
            ):
                raise TypeError(
                    "`batch` must use an integer dtype."
                )

            batch = batch.to(
                device=positions.device,
                dtype=torch.long,
            )

            if batch.numel() != n_atoms:
                raise ValueError(
                    "`batch` must contain one system index per atom."
                )

            if batch.numel() > 0:
                if int(batch[0].item()) != 0:
                    raise ValueError(
                        "`batch` system indices must start from zero."
                    )

                if torch.any(
                    batch[1:] < batch[:-1]
                ):
                    raise ValueError(
                        "PETBackbone requires atoms to be grouped by "
                        "system in the batched graph."
                    )

                unique_batch = torch.unique_consecutive(
                    batch,
                )

                expected_batch = torch.arange(
                    unique_batch.numel(),
                    device=batch.device,
                    dtype=batch.dtype,
                )

                if not torch.equal(
                    unique_batch,
                    expected_batch,
                ):
                    raise ValueError(
                        "`batch` system indices must be consecutive."
                    )

                if unique_batch.numel() != n_systems:
                    raise ValueError(
                        "`batch` and `ptr` describe different numbers "
                        "of systems."
                    )

                observed_counts = torch.bincount(
                    batch,
                    minlength=n_systems,
                )

                expected_counts = (
                    ptr[1:]
                    - ptr[:-1]
                ).to(
                    device=batch.device,
                    dtype=torch.long,
                )

                if not torch.equal(
                    observed_counts,
                    expected_counts,
                ):
                    raise ValueError(
                        "`batch` and `ptr` contain inconsistent atom "
                        "assignments."
                    )

            atom_systems = batch

        else:
            counts = (
                ptr[1:]
                - ptr[:-1]
            ).to(
                dtype=torch.long,
            )

            atom_systems = torch.repeat_interleave(
                torch.arange(
                    n_systems,
                    device=positions.device,
                    dtype=torch.long,
                ),
                counts,
            )

        # Cross-system graph edges cannot be represented in an individual
        # metatomic System.
        if edge_index.numel() > 0:
            edge_index_local = edge_index.to(
                device=positions.device,
                dtype=torch.long,
            )

            first_system = atom_systems[
                edge_index_local[0]
            ]

            second_system = atom_systems[
                edge_index_local[1]
            ]

            if torch.any(
                first_system != second_system
            ):
                raise ValueError(
                    "`edge_index` contains edges connecting atoms from "
                    "different systems."
                )

        cells = self._prepare_cells(
            data=data,
            cell=cell,
            n_systems=n_systems,
            positions=positions,
        )

        self._prepare_pbc(
            data=data,
            cells=cells,
            n_systems=n_systems,
        )