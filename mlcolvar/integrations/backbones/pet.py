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
    from metatomic.torch import ModelOutput, System

    _METATOMIC_AVAILABLE = True
    _METATOMIC_IMPORT_ERROR = None

except ImportError as exc:
    # Keep mlcolvar importable when the optional PET dependencies
    # are not installed.
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
    torch.float16: "float16",
    torch.float32: "float32",
    torch.float64: "float64",
    torch.bfloat16: "bfloat16",
}


def _infer_model_precision(
    model: nn.Module,
) -> Tuple[str, torch.dtype]:
    """Infer and validate the floating-point precision of a PET model."""

    floating_dtypes = {
        parameter.dtype
        for parameter in model.parameters()
        if parameter.is_floating_point()
    }

    floating_dtypes.update(
        buffer.dtype
        for buffer in model.buffers()
        if buffer.is_floating_point()
    )

    if len(floating_dtypes) == 0:
        raise ValueError(
            "Could not infer the floating-point precision of the PET model."
        )

    if len(floating_dtypes) != 1:
        raise ValueError(
            "The PET model contains mixed floating-point precisions: "
            f"{sorted(str(dtype) for dtype in floating_dtypes)}."
        )

    model_dtype = next(iter(floating_dtypes))

    if model_dtype not in _DTYPE_TO_PRECISION:
        raise ValueError(
            "Unsupported PET floating-point dtype: "
            f"{model_dtype}."
        )

    return (
        _DTYPE_TO_PRECISION[model_dtype],
        model_dtype,
    )


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
        hasattr(model, name)
        for name in required_attributes
    ):
        return False

    try:
        outputs = model.supported_outputs()

    except Exception:
        return False

    return "feature" in outputs


def _unwrap_pet_model(
    model: nn.Module,
) -> nn.Module:
    """Extract the native PET model from an optional wrapper.

    PET-MAD checkpoints can be wrapped by models such as
    ``LLPRUncertaintyModel``. In that case, the native PET model is
    available as ``wrapper.model``.
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
    """Infer the dimension of PET's public ``feature`` output."""

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

    return (
        d_node,
        d_pet,
        num_readout_layers,
    )


class PETBackbone(BaseAtomisticBackbone):
    """Extract atom-level features from a pretrained metatrain PET model.

    This adapter uses PET's public metatomic interface:

    ``mlcolvar graph -> metatomic Systems -> PET "feature" output``

    The mlcolvar graph neighbor list is converted to the neighbor-list
    ``TensorBlock`` requested by PET. PET then performs its own input
    preprocessing and message passing.

    Pooling, masking, and parameter freezing are handled by
    :class:`mlcolvar.integrations.atomistic.AtomisticFeaturizer`.

    Parameters
    ----------
    model
        Native ``metatrain.pet.model.PET`` model or a wrapper containing
        the native PET model in ``model.model``. PET-MAD LLPR checkpoints
        are therefore accepted directly.
    buffer
        Additional environment buffer used during mlcolvar graph
        construction.
    long_range_cutoff
        Optional mlcolvar long-range graph cutoff. Long-range graph edges
        marked by ``edge_masks_lr`` are excluded from the PET neighbor
        list.
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

        if not isinstance(
            model,
            nn.Module,
        ):
            raise TypeError(
                "`model` must be a torch.nn.Module."
            )

        # PET-MAD checkpoints can be wrapped by LLPRUncertaintyModel.
        model = _unwrap_pet_model(
            model
        )

        (
            d_node,
            d_pet,
            num_readout_layers,
        ) = _infer_pet_layout(
            model
        )

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

        neighbor_options = (
            requested_neighbor_lists[0]
        )

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

        if abs(
            model_cutoff
            - neighbor_cutoff
        ) > 1e-12:
            raise ValueError(
                "PET model cutoff and requested neighbor-list cutoff "
                "do not match: "
                f"model.cutoff={model_cutoff}, "
                f"neighbor cutoff={neighbor_cutoff}."
            )

        # PET concatenates:
        #
        # - one d_node node-feature block per readout layer;
        # - one d_pet cutoff-weighted edge-feature block per layer.
        out_features = (
            num_readout_layers
            * (
                d_node
                + d_pet
            )
        )

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

        self.num_readout_layers = (
            num_readout_layers
        )

        self.neighbor_cutoff = (
            neighbor_cutoff
        )

        self.neighbor_full_list = (
            neighbor_full_list
        )

        self.neighbor_strict = (
            neighbor_strict
        )

        (
            model_precision,
            model_dtype,
        ) = _infer_model_precision(
            model
        )

        self.model_precision = (
            model_precision
        )

        self.register_buffer(
            "_model_dtype_reference",
            torch.empty(
                0,
                dtype=model_dtype,
            ),
            persistent=False,
        )

        # Register the native PET model only after nn.Module has been
        # initialized by BaseAtomisticBackbone.
        self.model = model

        # Restore the checkpoint precision in case the supplied PET model
        # was already converted before wrapping.
        self._restore_model_precision()

    def _model_dtype(
        self,
    ) -> torch.dtype:
        """Return the native floating-point dtype of the PET model."""

        return _PRECISION_TO_DTYPE[
            self.model_precision
        ]

    def _restore_model_precision(
        self,
    ) -> None:
        """Restore PET parameters and buffers to checkpoint precision."""

        if not hasattr(
            self,
            "model",
        ):
            return

        model_dtype = (
            self._model_dtype()
        )

        self.model.to(
            dtype=model_dtype
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
                self
                ._model_dtype_reference
                .to(
                    device=model_device,
                    dtype=model_dtype,
                )
            )

    def _apply(
        self,
        fn,
    ):
        """Apply transforms while preserving PET checkpoint precision.

        Calling ``float()``, ``double()`` or ``to(dtype=...)`` on a
        surrounding mlcolvar model recursively reaches this backbone.
        PET is restored to its original checkpoint precision afterwards.
        Device moves are retained.
        """

        module = super()._apply(
            fn
        )

        if hasattr(
            self,
            "model",
        ):
            model_dtype = (
                self._model_dtype()
            )

            self._model_dtype_reference = (
                self
                ._model_dtype_reference
                .to(
                    dtype=model_dtype,
                )
            )

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

        input_positions = data[
            "positions"
        ]

        output_dtype = (
            input_positions.dtype
        )

        output_device = (
            input_positions.device
        )

        model_dtype = (
            self._model_dtype()
        )

        model_device = (
            self._model_dtype_reference.device
        )

        # PET-MAD is trained and exported in a fixed precision, commonly
        # float32. Convert graph geometry to this internal precision while
        # preserving the autograd connection to the original positions.
        positions = input_positions.to(
            device=model_device,
            dtype=model_dtype,
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

        atomic_numbers = self.atomic_numbers.to(
            device=positions.device,
        )

        species_indices = node_attrs.argmax(
            dim=-1
        ).to(
            device=positions.device,
            dtype=torch.long,
        )

        atom_types = atomic_numbers[
            species_indices
        ].to(
            dtype=torch.int32,
        )

        neighbor_options = (
            self.model
            .requested_neighbor_lists()[0]
        )

        systems: List[System] = []

        for system_index in range(
            n_systems
        ):
            start = int(
                ptr[system_index].item()
            )

            end = int(
                ptr[system_index + 1].item()
            )

            system_positions = positions[
                start:end
            ]

            system_types = atom_types[
                start:end
            ]

            system_cell = cells[
                system_index
            ]

            system_pbc = pbc[
                system_index
            ]

            system = System(
                types=system_types,
                positions=system_positions,
                cell=system_cell,
                pbc=system_pbc,
            )

            neighbors = self._build_neighbor_list(
                data=data,
                positions=system_positions,
                cell=system_cell,
                start=start,
                end=end,
            )

            system.add_neighbor_list(
                neighbor_options,
                neighbors,
            )

            systems.append(
                system
            )

        requested_outputs: Dict[
            str,
            ModelOutput,
        ] = {
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
            if features.dim() != 2:
                raise RuntimeError(
                    "Expected PET `feature` values to be rank 2, "
                    "with shape [n_atoms, n_features]."
                )

            if (
                features.size(0)
                != positions.size(0)
            ):
                raise RuntimeError(
                    "The number of PET feature rows does not match "
                    "the number of graph atoms. Expected "
                    f"{positions.size(0)}, found "
                    f"{features.size(0)}."
                )

            if (
                features.size(1)
                != self.out_features
            ):
                raise RuntimeError(
                    "Unexpected PET feature dimension. Expected "
                    f"{self.out_features}, found "
                    f"{features.size(1)}."
                )

        # Restore the dtype/device expected by the surrounding mlcolvar
        # graph and readout. The cast preserves coordinate gradients.
        return features.to(
            device=output_device,
            dtype=output_dtype,
        )

    def _build_neighbor_list(
        self,
        data: Dict[str, torch.Tensor],
        positions: torch.Tensor,
        cell: torch.Tensor,
        start: int,
        end: int,
    ) -> TensorBlock:
        """Convert one mlcolvar graph into a metatomic neighbor list."""

        edge_index = data[
            "edge_index"
        ].to(
            device=positions.device,
            dtype=torch.long,
        )

        unit_shifts = data[
            "unit_shifts"
        ].to(
            device=positions.device,
        )

        first_global = edge_index[
            0
        ]

        second_global = edge_index[
            1
        ]

        edge_mask = (
            (first_global >= start)
            & (first_global < end)
            & (second_global >= start)
            & (second_global < end)
        )

        # mlcolvar can append a separate long-range edge list to the
        # short-range graph. PET should only receive its own requested
        # short-range neighbor list.
        if "edge_masks_lr" in data:
            long_range_mask = (
                data["edge_masks_lr"]
                .reshape(-1)
                .to(
                    device=edge_mask.device,
                    dtype=torch.bool,
                )
            )

            edge_mask = (
                edge_mask
                & ~long_range_mask
            )

        first_atom = (
            first_global[
                edge_mask
            ]
            - start
        )

        second_atom = (
            second_global[
                edge_mask
            ]
            - start
        )

        cell_shifts = torch.round(
            unit_shifts[
                edge_mask
            ]
        ).to(
            dtype=torch.int32,
        )

        cartesian_shifts = (
            cell_shifts.to(
                dtype=positions.dtype
            )
            @ cell
        )

        edge_vectors = (
            positions[
                second_atom
            ]
            - positions[
                first_atom
            ]
            + cartesian_shifts
        )

        # PET currently requests a strict neighbor list. Filtering here
        # also protects against graph buffers or a larger mlcolvar cutoff.
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
                    dtype=torch.int32
                ).reshape(-1, 1),
                second_atom.to(
                    dtype=torch.int32
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

        components = [
            Labels(
                names=["xyz"],
                values=torch.arange(
                    3,
                    device=positions.device,
                    dtype=torch.int32,
                ).reshape(-1, 1),
            )
        ]

        properties = Labels(
            names=["distance"],
            values=torch.zeros(
                (
                    1,
                    1,
                ),
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
    def _validate_graph_input(
        self,
        data: Dict[str, torch.Tensor],
        cell: Optional[torch.Tensor],
    ) -> None:
        """Validate graph fields required by the PET adapter."""

        required_keys = [
            "positions",
            "node_attrs",
            "edge_index",
            "unit_shifts",
        ]

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

        positions = data[
            "positions"
        ]

        node_attrs = data[
            "node_attrs"
        ]

        edge_index = data[
            "edge_index"
        ]

        unit_shifts = data[
            "unit_shifts"
        ]

        if (
            positions.dim() != 2
            or positions.size(1) != 3
        ):
            raise ValueError(
                "`positions` must have shape [n_atoms, 3], "
                f"found {tuple(positions.shape)}."
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
                "The width of `node_attrs` does not match the PET "
                "atomic-type table. Align the dataset with "
                "`AtomisticFeaturizer.align_dataset()` before "
                "constructing the datamodule."
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
            unit_shifts.dim() != 2
            or unit_shifts.size(1) != 3
        ):
            raise ValueError(
                "`unit_shifts` must have shape [n_edges, 3], "
                f"found {tuple(unit_shifts.shape)}."
            )

        if (
            unit_shifts.size(0)
            != edge_index.size(1)
        ):
            raise ValueError(
                "`unit_shifts` and `edge_index` contain different "
                "numbers of edges."
            )

        if not torch.allclose(
            unit_shifts,
            torch.round(
                unit_shifts
            ),
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
            ) != positions.size(0)
        ):
            raise ValueError(
                "Invalid graph `ptr`/`batch` information."
            )

        if "batch" in data:
            batch = data[
                "batch"
            ].to(
                dtype=torch.long
            )

            if batch.numel() != positions.size(0):
                raise ValueError(
                    "`batch` must contain one system index per atom."
                )

            if batch.numel() > 1:
                if torch.any(
                    batch[1:]
                    < batch[:-1]
                ):
                    raise ValueError(
                        "PETBackbone requires atoms to be grouped by "
                        "system in the batched graph."
                    )

        # Validate the optional/runtime cell shape early.
        self._prepare_cells(
            data=data,
            cell=cell,
            n_systems=ptr.numel() - 1,
            positions=positions,
        )