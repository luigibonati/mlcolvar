from collections.abc import Sequence
from copy import deepcopy
from pathlib import Path
from typing import Optional

import torch
from torch import nn

from mlcolvar.core.nn.graph import SchNetModel
from mlcolvar.representation import MACERepresentation, MLColvarRepresentation
from mlcolvar.representation._utils import _as_atomic_number_list, as_float, as_positive_int
from mlcolvar.utils.evaluation import module_reference_tensor
from ._compat import (
    MetatomicAtomisticModel, ModelCapabilities, ModelMetadata,
    ModelOutput, NeighborListOptions,
)
from .wrapper import _CVInferenceModel, _MetatomicCVWrapper

__all__ = ["create_metatomic_model", "export_metatomic_model"]


def _network(model: nn.Module) -> nn.Module:
    """Return the module consuming atomistic graph inputs."""
    preprocessing = getattr(model, "preprocessing", None)
    if getattr(preprocessing, "input_kind", None) == "graph":
        return preprocessing
    return getattr(model, "nn", model)


def _model_attribute(model: nn.Module, network: nn.Module, name: str):
    value = getattr(model, name, None)
    return getattr(network, name, None) if value is None else value


def _get_neighbor_options(
    network: nn.Module, short_cutoff: float, long_cutoff: float | None,
    buffer: float = 0.0,
) -> tuple[NeighborListOptions, Optional[NeighborListOptions]]:
    """Use the network's declared requests when compatible; otherwise create them."""
    requested = []
    for module in network.modules():
        request = getattr(module, "requested_neighbor_lists", None)
        if callable(request):
            requested.extend(request())
        else:
            option = getattr(module, "neighbor_options", None)
            if option is not None:
                requested.append(option)

    # Reuse the long list for buffered environment discovery when it covers
    # short_cutoff + buffer. This avoids requesting a third neighbor list.
    use_long_buffer = (long_cutoff is not None and buffer > 0 and
                       long_cutoff >= short_cutoff + buffer)
    short_radius = short_cutoff if use_long_buffer else short_cutoff + buffer
    def match(radius):
        matches = [opt for opt in requested if abs(opt.cutoff - radius) < 1e-6]
        if len(matches) > 1 and any(
            opt.full_list != matches[0].full_list or opt.strict != matches[0].strict
            for opt in matches[1:]
        ):
            raise ValueError(f"Conflicting neighbor lists at cutoff {radius}.")
        return matches[0] if matches else None

    short = match(short_radius)
    long = match(long_cutoff) if long_cutoff is not None else None
    unused = [opt for opt in requested if
              abs(opt.cutoff - short_radius) >= 1e-6 and
              (long_cutoff is None or abs(opt.cutoff - long_cutoff) >= 1e-6)]
    # A model may independently request its native short list when we need
    # an extended short list for environment discovery. Metatomic will handle both.
    if unused and not (buffer > 0 and all(abs(opt.cutoff - short_cutoff) < 1e-6 for opt in unused)):
        raise ValueError("Unsupported neighbor-list requests; an explicit adapter is required.")

    if short is None:
        short = NeighborListOptions(
            cutoff=short_radius, full_list=True, strict=True, requestor="mlcolvar short",
        )
    if long_cutoff is not None and long is None:
        long = NeighborListOptions(
            cutoff=long_cutoff, full_list=True, strict=True, requestor="mlcolvar long",
        )
    if not short.full_list or not short.strict or (long is not None and (not long.full_list or not long.strict)):
        raise ValueError("The mlcolvar graph adapter requires strict, full neighbor lists.")
    return short, long


def _message_passing_steps(network: nn.Module) -> int | None:
    """Return the known message-passing depth; None for unknown architectures."""
    if isinstance(network, MLColvarRepresentation):
        network = network.encoder
    if isinstance(network, SchNetModel):
        return len(network.layers)
    if isinstance(network, MACERepresentation):
        return as_positive_int(network.model.num_interactions, "num_interactions")
    return None


def _resolve_interaction_range(
    network: nn.Module, interaction_range: float | None,
    short_cutoff: float, long_cutoff: float | None, buffer: float,
) -> float:
    """Calculate a conservative spatial dependency range for Metatomic."""
    cutoff = max(short_cutoff, long_cutoff or 0.0)
    minimum = max(cutoff, short_cutoff + buffer)
    steps = _message_passing_steps(network)
    if steps is not None:
        minimum = max(minimum, steps * cutoff)
    if interaction_range is None:
        # Unknown architectures may involve non-local operations. Never silently
        # advertise only a single neighbor-list cutoff as their receptive field.
        return minimum if steps is not None else float("inf")
    interaction_range = as_float(interaction_range, "interaction_range")
    if not interaction_range >= minimum:
        raise ValueError(
            f"`interaction_range` must be at least {minimum:g} for this model."
        )
    return interaction_range


def _prepare_network(network: nn.Module) -> nn.Module:
    for module in network.modules():
        prepare = getattr(module, "prepare_for_torchscript", None)
        if callable(prepare):
            prepare()
    return network


def _build_postprocessing(model: nn.Module, network: nn.Module) -> nn.Module:
    if network is model:
        return nn.Identity()
    modules = []
    include = network is getattr(model, "preprocessing", None)
    for name in getattr(model, "BLOCKS", ()):
        block = getattr(model, name, None)
        if block is None:
            continue
        if not include:
            if block is network:
                include = True
            continue
        modules.append(block)
    postprocessing = getattr(model, "postprocessing", None)
    if postprocessing is not None:
        modules.append(postprocessing)
    return nn.Sequential(*modules) if modules else nn.Identity()


def _make_inference_model(
    model: nn.Module, atomic_types: Sequence[int], short_cutoff: float,
    long_cutoff: float | None, graph_mode: str,
    system_indices: Sequence[int] | None, environment_indices: Sequence[int] | None,
    subsystem_indices: Sequence[int] | None, subsystem_atomic_numbers: Sequence[int] | None,
    buffer: float,
) -> _CVInferenceModel:
    network = _network(model)
    short, long = _get_neighbor_options(
        network, short_cutoff, long_cutoff, buffer if system_indices is not None else 0.0,
    )
    components = deepcopy(nn.ModuleList([network, _build_postprocessing(model, network)])).eval()
    return _CVInferenceModel(
        network=_prepare_network(components[0]), postprocessing=components[1],
        atomic_numbers=torch.as_tensor(atomic_types, dtype=torch.long),
        neighbor_options=short, long_neighbor_options=long, short_cutoff=short_cutoff,
        long_cutoff=long_cutoff, graph_mode=graph_mode,
        system_indices=list(system_indices) if system_indices is not None else None,
        environment_indices=list(environment_indices) if environment_indices is not None else None,
        subsystem_indices=list(subsystem_indices) if subsystem_indices is not None else None,
        subsystem_atomic_numbers=(list(subsystem_atomic_numbers) if subsystem_atomic_numbers is not None else None),
        buffer=buffer if system_indices is not None else 0.0,
    ).eval()


def create_metatomic_model(
    model: nn.Module,
    out_features: int | None = None,
    atomic_types: Sequence[int] | None = None,
    interaction_range: float | None = None,
    *,
    sample_kind: str | None = None,
    long_range_cutoff: float | None = None,
    graph_mode: str = "merged",
    system_indices: Sequence[int] | None = None,
    environment_indices: Sequence[int] | None = None,
    subsystem_indices: Sequence[int] | None = None,
    subsystem_atomic_numbers: Sequence[int] | None = None,
    length_unit: str | None = None,
    dtype: str | None = None,
    supported_devices: Sequence[str] | None = None,
    name: str = "mlcolvar collective variable",
    description: str = "Collective variable model exported from mlcolvar.",
    authors: Sequence[str] | None = None,
) -> MetatomicAtomisticModel:
    """Export graph representations with optional dual cutoffs and atom selection.

    ``system_indices`` define core atoms. Other atoms form the environment unless
    ``environment_indices`` restricts the candidate pool. Environment membership
    is recomputed from distances at every forward call using cutoff + buffer.
    ``subsystem_indices``/``subsystem_atomic_numbers`` define long-range endpoints.
    Indices are LOCAL to each System and require consistent atom ordering.
    ``interaction_range`` is inferred for SchNet and MACE. For unknown models,
    it defaults conservatively to infinity; provide it explicitly if known.
    """
    network = _network(model)
    if sample_kind is None:
        sample_kind = getattr(model, "output_kind", "system")
    if sample_kind not in {"system", "atom"}:
        raise ValueError("`sample_kind` must be 'system' or 'atom'.")
    if graph_mode not in {"merged", "separate"}:
        raise ValueError("`graph_mode` must be 'merged' or 'separate'.")
    if graph_mode == "separate" and (system_indices is not None or environment_indices is not None):
        raise ValueError("Dynamic environment selection is only supported in merged mode.")

    if out_features is None:
        out_features = getattr(model, "n_cvs", None)
        if out_features is None:
            out_features = getattr(network, "out_features", None)
        if out_features is None:
            raise ValueError("Could not infer `out_features`.")
    if atomic_types is None:
        atomic_types = _model_attribute(model, network, "atomic_numbers")
        if atomic_types is None:
            raise ValueError("Could not infer `atomic_types`.")

    cutoff_value = _model_attribute(model, network, "cutoff")
    if cutoff_value is None:
        raise ValueError("The graph model must provide its short-range `cutoff`.")
    short_cutoff = as_float(cutoff_value, "cutoff")
    if short_cutoff <= 0:
        raise ValueError("`cutoff` must be positive.")
    if long_range_cutoff is None:
        long_range_cutoff = _model_attribute(model, network, "long_range_cutoff")
    long_cutoff = None if long_range_cutoff is None else as_float(long_range_cutoff, "long_range_cutoff")
    if long_cutoff is not None and long_cutoff <= 0:
        long_cutoff = None
    if long_cutoff is not None and long_cutoff <= short_cutoff:
        raise ValueError("`long_range_cutoff` must be larger than `cutoff`.")
    if graph_mode == "separate" and long_cutoff is None:
        raise ValueError("Separate mode requires `long_range_cutoff`.")
    if long_cutoff is not None and graph_mode == "merged" and subsystem_indices is None and subsystem_atomic_numbers is None:
        raise ValueError("Merged dual-cutoff graphs require an explicit subsystem selection.")
    if subsystem_indices is not None and subsystem_atomic_numbers is not None:
        raise ValueError("Specify either `subsystem_indices` or `subsystem_atomic_numbers`.")
    if environment_indices is not None and system_indices is None:
        raise ValueError("`environment_indices` requires `system_indices`.")

    buffer_value = _model_attribute(model, network, "buffer")
    buffer = 0.0 if buffer_value is None else as_float(buffer_value, "buffer")
    if buffer < 0:
        raise ValueError("`buffer` must be non-negative.")
    out_features = as_positive_int(out_features, "out_features")
    atomic_types = _as_atomic_number_list(atomic_types, "atomic_types")
    interaction_range = _resolve_interaction_range(
        network, interaction_range, short_cutoff, long_cutoff,
        buffer if system_indices is not None else 0.0,
    )
    if length_unit is None:
        length_unit = _model_attribute(model, network, "length_unit") or "angstrom"
    if dtype is None:
        reference = module_reference_tensor(model)
        dtype = "float64" if reference.dtype == torch.float64 else "float32"
    if dtype not in {"float32", "float64"}:
        raise ValueError("`dtype` must be 'float32' or 'float64'.")
    supported_devices = ("cpu", "cuda") if supported_devices is None else supported_devices
    authors = () if authors is None else authors

    wrapper = _MetatomicCVWrapper(
        model=_make_inference_model(
            model, atomic_types, short_cutoff, long_cutoff, graph_mode,
            system_indices, environment_indices, subsystem_indices, subsystem_atomic_numbers, buffer,
        ),
        out_features=out_features, sample_kind=sample_kind,
    ).eval()
    return MetatomicAtomisticModel(
        module=wrapper,
        metadata=ModelMetadata(name=name, description=description, authors=list(authors)),
        capabilities=ModelCapabilities(
            length_unit=length_unit, outputs={"feature": ModelOutput(sample_kind=sample_kind)},
            atomic_types=atomic_types, interaction_range=interaction_range,
            supported_devices=list(supported_devices), dtype=dtype,
        ),
    )


def export_metatomic_model(
    model: nn.Module, path: str | Path,
    out_features: int | None = None,
    atomic_types: Sequence[int] | None = None,
    interaction_range: float | None = None,
    *,
    sample_kind: str | None = None,
    long_range_cutoff: float | None = None,
    graph_mode: str = "merged",
    system_indices: Sequence[int] | None = None,
    environment_indices: Sequence[int] | None = None,
    subsystem_indices: Sequence[int] | None = None,
    subsystem_atomic_numbers: Sequence[int] | None = None,
    length_unit: str | None = None,
    dtype: str | None = None,
    supported_devices: Sequence[str] | None = None,
    name: str = "mlcolvar collective variable",
    description: str = "Collective variable model exported from mlcolvar.",
    authors: Sequence[str] | None = None,
    collect_extensions: str | Path | None = None,
) -> Path:
    """Create and save a Metatomic model."""
    path = Path(path)
    if path.suffix != ".pt":
        raise ValueError("The exported Metatomic model must use the '.pt' extension.")
    path.parent.mkdir(parents=True, exist_ok=True)
    metatomic_model = create_metatomic_model(
        model=model, out_features=out_features, atomic_types=atomic_types,
        interaction_range=interaction_range, sample_kind=sample_kind,
        long_range_cutoff=long_range_cutoff, graph_mode=graph_mode,
        system_indices=system_indices, environment_indices=environment_indices,
        subsystem_indices=subsystem_indices, subsystem_atomic_numbers=subsystem_atomic_numbers,
        length_unit=length_unit, dtype=dtype, supported_devices=supported_devices,
        name=name, description=description, authors=authors,
    )
    if collect_extensions is None:
        metatomic_model.save(str(path))
    else:
        extensions = Path(collect_extensions)
        extensions.mkdir(parents=True, exist_ok=True)
        metatomic_model.save(str(path), collect_extensions=str(extensions))
    return path
