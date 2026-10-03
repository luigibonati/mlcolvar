from collections.abc import Sequence
from copy import deepcopy
from pathlib import Path

import torch
from torch import nn

from mlcolvar.representation._utils import (
    _as_atomic_number_list,
    as_float,
    as_positive_int,
    module_reference_tensor,
)
from ._compat import (
    MetatomicAtomisticModel,
    ModelCapabilities,
    ModelMetadata,
    ModelOutput,
    NeighborListOptions,
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
    """Get export metadata from the model or atomistic network."""
    value = getattr(model, name, None)
    return getattr(network, name, None) if value is None else value


def _get_neighbor_options(
    network: nn.Module,
    interaction_range: float,
) -> NeighborListOptions:
    """Return neighbor-list options required by the network."""
    for module in network.modules():
        request = getattr(module, "requested_neighbor_lists", None)
        if callable(request):
            options = request()
            if len(options) == 1:
                return options[0]
            if len(options) > 1:
                raise ValueError(
                    "The model requests multiple neighbor lists; "
                    "only one is currently supported."
                )

    for module in network.modules():
        options = getattr(module, "neighbor_options", None)
        if options is not None:
            return options

    return NeighborListOptions(
        cutoff=interaction_range,
        full_list=True,
        strict=True,
        requestor="mlcolvar Metatomic adapter",
    )


def _prepare_network(network: nn.Module) -> nn.Module:
    """Prepare an atomistic network for TorchScript export."""
    network = deepcopy(network).eval()
    for module in network.modules():
        prepare = getattr(module, "prepare_for_torchscript", None)
        if callable(prepare):
            prepare()
    return network


def _build_postprocessing(model: nn.Module, network: nn.Module) -> nn.Module:
    """Build the inference pipeline applied after the graph network."""
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
        modules.append(deepcopy(block))

    postprocessing = getattr(model, "postprocessing", None)
    if postprocessing is not None:
        modules.append(deepcopy(postprocessing))
    return nn.Sequential(*modules).eval() if modules else nn.Identity()


def _make_inference_model(
    model: nn.Module,
    atomic_types: Sequence[int],
    interaction_range: float,
) -> _CVInferenceModel:
    """Build the complete Metatomic inference pipeline."""
    network = _network(model)
    return _CVInferenceModel(
        network=_prepare_network(network),
        postprocessing=_build_postprocessing(model, network),
        atomic_numbers=torch.as_tensor(atomic_types, dtype=torch.long),
        neighbor_options=_get_neighbor_options(network, interaction_range),
    ).eval()


def create_metatomic_model(
    model: nn.Module,
    out_features: int | None = None,
    atomic_types: Sequence[int] | None = None,
    interaction_range: float | None = None,
    *,
    length_unit: str | None = None,
    dtype: str | None = None,
    supported_devices: Sequence[str] | None = None,
    name: str = "mlcolvar collective variable",
    description: str = "Collective variable model exported from mlcolvar.",
    authors: Sequence[str] | None = None,
) -> MetatomicAtomisticModel:
    """Create an exportable Metatomic model."""
    network = _network(model)

    if out_features is None:
        out_features = getattr(model, "n_cvs", None)
        if out_features is None:
            out_features = getattr(network, "out_features", None)
        if out_features is None:
            raise ValueError(
                "Could not infer `out_features`; provide it explicitly."
            )

    if atomic_types is None:
        atomic_types = _model_attribute(model, network, "atomic_numbers")
        if atomic_types is None:
            raise ValueError(
                "Could not infer `atomic_types`; provide them explicitly."
            )

    if interaction_range is None:
        interaction_range = _model_attribute(model, network, "cutoff")
        if interaction_range is None:
            raise ValueError(
                "Could not infer `interaction_range`; provide it explicitly."
            )

    if length_unit is None:
        length_unit = _model_attribute(model, network, "length_unit") or "angstrom"

    if dtype is None:
        reference = module_reference_tensor(model)
        dtype = "float64" if reference.dtype == torch.float64 else "float32"

    out_features = as_positive_int(out_features, "out_features")
    atomic_types = _as_atomic_number_list(atomic_types, "atomic_types")
    interaction_range = as_float(interaction_range, "interaction_range")

    if interaction_range <= 0:
        raise ValueError("`interaction_range` must be positive.")
    if dtype not in {"float32", "float64"}:
        raise ValueError("`dtype` must be 'float32' or 'float64'.")

    supported_devices = (
        ("cpu", "cuda") if supported_devices is None else supported_devices
    )
    authors = () if authors is None else authors

    wrapper = _MetatomicCVWrapper(
        model=_make_inference_model(
            model,
            atomic_types,
            interaction_range,
        ),
        out_features=out_features,
    ).eval()

    return MetatomicAtomisticModel(
        module=wrapper,
        metadata=ModelMetadata(
            name=name,
            description=description,
            authors=list(authors),
        ),
        capabilities=ModelCapabilities(
            length_unit=length_unit,
            outputs={"feature": ModelOutput(sample_kind="system")},
            atomic_types=atomic_types,
            interaction_range=interaction_range,
            supported_devices=list(supported_devices),
            dtype=dtype,
        ),
    )


def export_metatomic_model(
    model: nn.Module,
    path: str | Path,
    out_features: int | None = None,
    atomic_types: Sequence[int] | None = None,
    interaction_range: float | None = None,
    *,
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
        model=model,
        out_features=out_features,
        atomic_types=atomic_types,
        interaction_range=interaction_range,
        length_unit=length_unit,
        dtype=dtype,
        supported_devices=supported_devices,
        name=name,
        description=description,
        authors=authors,
    )

    if collect_extensions is None:
        metatomic_model.save(str(path))
    else:
        extensions = Path(collect_extensions)
        extensions.mkdir(parents=True, exist_ok=True)
        metatomic_model.save(
            str(path),
            collect_extensions=str(extensions),
        )
    return path