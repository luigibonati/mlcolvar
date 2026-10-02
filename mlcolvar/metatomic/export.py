import copy
from pathlib import Path
from typing import Sequence

import torch
from torch import nn

from ._compat import (
    MetatomicAtomisticModel,
    ModelCapabilities,
    ModelMetadata,
    ModelOutput,
    NeighborListOptions,
)
from .wrapper import CVInferenceModel, MetatomicCVWrapper

__all__ = [
    "create_metatomic_model",
    "export_metatomic_model",
]


def _as_float(value) -> float:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().item()
    return float(value)


def _as_int(value) -> int:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().item()
    return int(value)


def _network(model: nn.Module) -> nn.Module:
    """Return the module that consumes atomistic graph inputs."""
    preprocessing = getattr(model, "preprocessing", None)
    if getattr(preprocessing, "input_kind", None) == "graph":
        return preprocessing

    network = getattr(model, "nn", None)
    return model if network is None else network


def _model_attribute(model: nn.Module, network: nn.Module, name: str):
    """Get export metadata from the complete model or atomistic network."""
    value = getattr(model, name, None)
    if value is None and network is not model:
        value = getattr(network, name, None)
    return value


def _reference_tensor(module: nn.Module) -> torch.Tensor:
    """Return a floating-point tensor matching a module dtype and device."""
    for tensor in module.parameters():
        if tensor.is_floating_point() or tensor.is_complex():
            return tensor

    for tensor in module.buffers():
        if tensor.is_floating_point() or tensor.is_complex():
            return tensor

    return torch.empty(())


def _get_neighbor_options(
    network: nn.Module,
    interaction_range: float,
) -> NeighborListOptions:
    """Return neighbor-list options required by the atomistic network."""
    for module in network.modules():
        request = getattr(module, "requested_neighbor_lists", None)
        if request is None:
            continue

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
        cutoff=float(interaction_range),
        full_list=True,
        strict=True,
        requestor="mlcolvar Metatomic adapter",
    )


def _prepare_network(network: nn.Module) -> nn.Module:
    """Prepare an atomistic network for TorchScript export."""
    network = copy.deepcopy(network).eval()

    for module in list(network.modules()):
        prepare = getattr(module, "prepare_for_torchscript", None)
        if prepare is None:
            continue

        try:
            prepare()
        except Exception as exc:
            raise RuntimeError(
                f"Failed to prepare {module.__class__.__name__} "
                "for TorchScript export."
            ) from exc

    return network


def _build_postprocessing(
    model: nn.Module,
    network: nn.Module,
) -> nn.Module:
    """Build the pure inference pipeline applied after the graph network."""
    if network is model:
        return nn.Identity()

    preprocessing = getattr(model, "preprocessing", None)
    blocks = getattr(model, "BLOCKS", ())
    modules = []

    # If the graph network is preprocessing, all CV blocks act downstream.
    include = network is preprocessing

    for name in blocks:
        block = getattr(model, name, None)
        if block is None:
            continue

        # For legacy graph CVs, skip the graph block itself and include only
        # subsequent blocks such as a committor sigmoid.
        if not include:
            if block is network:
                include = True
            continue

        modules.append(copy.deepcopy(block))

    postprocessing = getattr(model, "postprocessing", None)
    if postprocessing is not None:
        modules.append(copy.deepcopy(postprocessing))

    if not modules:
        return nn.Identity()

    return nn.Sequential(*modules).eval()


def _make_inference_model(
    model: nn.Module,
    atomic_types: Sequence[int],
    interaction_range: float,
) -> CVInferenceModel:
    """Build the complete Metatomic inference pipeline."""
    model = model.eval()
    network = _network(model)
    postprocessing = _build_postprocessing(model, network)

    neighbor_options = _get_neighbor_options(
        network,
        interaction_range,
    )

    return CVInferenceModel(
        network=_prepare_network(network),
        postprocessing=postprocessing,
        atomic_numbers=torch.as_tensor(
            atomic_types,
            dtype=torch.long,
        ),
        neighbor_options=neighbor_options,
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
    """Create an exportable Metatomic model.

    Parameters
    ----------
    model : torch.nn.Module
        mlcolvar model to export. Atomistic graph input may be handled directly
        by the model or by graph-based preprocessing.
    out_features : int, optional
        Number of system-level output features. If omitted, infer it from
        ``model.n_cvs`` or the atomistic network.
    atomic_types : sequence of int, optional
        Supported atomic numbers. If omitted, infer them from ``atomic_numbers``.
    interaction_range : float, optional
        Neighbor-list cutoff. If omitted, infer it from ``cutoff``.
    length_unit : str, optional
        Length unit used by the model. Defaults to ``"angstrom"``.
    dtype : {"float32", "float64"}, optional
        Floating-point dtype exposed through Metatomic capabilities.
    supported_devices : sequence of str, optional
        Supported devices. Defaults to CPU and CUDA.
    name : str
        Exported model name.
    description : str
        Exported model description.
    authors : sequence of str, optional
        Model authors.

    Returns
    -------
    MetatomicAtomisticModel
        Exportable Metatomic model.
    """
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
        atomic_types = _model_attribute(
            model,
            network,
            "atomic_numbers",
        )
        if atomic_types is None:
            raise ValueError(
                "Could not infer `atomic_types`; provide them explicitly."
            )

    if interaction_range is None:
        interaction_range = _model_attribute(
            model,
            network,
            "cutoff",
        )
        if interaction_range is None:
            raise ValueError(
                "Could not infer `interaction_range`; provide it explicitly."
            )

    if length_unit is None:
        length_unit = _model_attribute(
            model,
            network,
            "length_unit",
        )
        if length_unit is None:
            length_unit = "angstrom"

    if dtype is None:
        reference = _reference_tensor(model)
        dtype = (
            "float64"
            if reference.dtype == torch.float64
            else "float32"
        )

    out_features = _as_int(out_features)
    atomic_types = (
        torch.as_tensor(
            atomic_types,
            dtype=torch.long,
        )
        .detach()
        .cpu()
        .reshape(-1)
        .tolist()
    )
    atomic_types = [int(value) for value in atomic_types]
    interaction_range = _as_float(interaction_range)

    if out_features <= 0:
        raise ValueError(
            "`out_features` must be positive."
        )
    if not atomic_types:
        raise ValueError(
            "`atomic_types` must contain at least one atomic type."
        )
    if interaction_range < 0:
        raise ValueError(
            "`interaction_range` must be non-negative."
        )
    if dtype not in {"float32", "float64"}:
        raise ValueError(
            "`dtype` must be 'float32' or 'float64'."
        )

    supported_devices = (
        ("cpu", "cuda")
        if supported_devices is None
        else supported_devices
    )
    authors = () if authors is None else authors

    inference = _make_inference_model(
        model,
        atomic_types,
        interaction_range,
    )

    wrapper = MetatomicCVWrapper(
        model=inference,
        out_features=out_features,
    ).eval()

    metadata = ModelMetadata(
        name=name,
        description=description,
        authors=list(authors),
    )

    capabilities = ModelCapabilities(
        length_unit=length_unit,
        outputs={
            "feature": ModelOutput(
                sample_kind="system",
            )
        },
        atomic_types=atomic_types,
        interaction_range=interaction_range,
        supported_devices=list(supported_devices),
        dtype=dtype,
    )

    return MetatomicAtomisticModel(
        module=wrapper,
        metadata=metadata,
        capabilities=capabilities,
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
    """Create and save a Metatomic model.

    Parameters
    ----------
    model : torch.nn.Module
        mlcolvar model to export.
    path : str or pathlib.Path
        Output path with ``.pt`` extension.
    out_features : int, optional
        Number of output features.
    atomic_types : sequence of int, optional
        Supported atomic numbers.
    interaction_range : float, optional
        Neighbor-list cutoff.
    length_unit : str, optional
        Length unit used by the model.
    dtype : {"float32", "float64"}, optional
        Floating-point dtype exposed by the model.
    supported_devices : sequence of str, optional
        Supported devices.
    name : str
        Exported model name.
    description : str
        Exported model description.
    authors : sequence of str, optional
        Model authors.
    collect_extensions : str or pathlib.Path, optional
        Directory used to collect external TorchScript extensions.

    Returns
    -------
    pathlib.Path
        Path of the saved Metatomic model.
    """
    path = Path(path)
    if path.suffix != ".pt":
        raise ValueError(
            "The exported Metatomic model must use the '.pt' extension."
        )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

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
        extensions.mkdir(
            parents=True,
            exist_ok=True,
        )
        metatomic_model.save(
            str(path),
            collect_extensions=str(extensions),
        )

    return path