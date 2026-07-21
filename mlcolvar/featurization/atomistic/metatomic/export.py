from pathlib import Path
from typing import Optional, Sequence, Union

import torch

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
    """Convert a scalar or scalar tensor to a Python float."""
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().item())

    return float(value)


def _get_neighbor_options(
    network: torch.nn.Module,
    interaction_range: float,
) -> NeighborListOptions:
    """Find or construct the neighbor-list options required by the model.

    Resolution order:

    1. ``requested_neighbor_lists()`` on backbone, featurizer, or network;
    2. a direct ``neighbor_options`` attribute;
    3. a ``cutoff`` attribute;
    4. the user-provided ``interaction_range``.

    The current adapter supports exactly one neighbor list.
    """
    featurizer = getattr(network, "featurizer", None)
    backbone = (
        getattr(featurizer, "backbone", None)
        if featurizer is not None
        else None
    )

    candidates = [backbone, featurizer, network]

    for module in candidates:
        if module is None:
            continue

        requested_neighbor_lists = getattr(
            module,
            "requested_neighbor_lists",
            None,
        )

        if requested_neighbor_lists is None:
            continue

        options = requested_neighbor_lists()

        if len(options) == 1:
            return options[0]

        if len(options) > 1:
            raise ValueError(
                "The atomistic model requests multiple neighbor lists, but "
                "the current mlcolvar Metatomic adapter supports exactly one."
            )

    for module in candidates:
        if module is None:
            continue

        neighbor_options = getattr(
            module,
            "neighbor_options",
            None,
        )

        if neighbor_options is not None:
            return neighbor_options

    cutoff = float(interaction_range)

    for module in candidates:
        if module is None:
            continue

        module_cutoff = getattr(module, "cutoff", None)

        if module_cutoff is not None:
            cutoff = _as_float(module_cutoff)
            break

    return NeighborListOptions(
        cutoff=cutoff,
        full_list=True,
        strict=True,
        requestor="mlcolvar Metatomic adapter",
    )


def _contains_e3nn_or_mace(
    module: torch.nn.Module,
) -> bool:
    """Return whether a module tree contains MACE/e3nn components."""
    for child in module.modules():
        module_name = child.__class__.__module__.lower()
        class_name = child.__class__.__name__.lower()

        if (
            "e3nn" in module_name
            or "mace" in module_name
            or "mace" in class_name
        ):
            return True

    return False


def _prepare_network(
    network: torch.nn.Module,
) -> torch.nn.Module:
    """Prepare MACE/e3nn networks for the outer Metatomic TorchScript pass.

    MACE contains e3nn modules that can not always be compiled correctly by a
    plain recursive ``torch.jit.script`` call. e3nn's JIT utility recursively
    applies the compile mode declared by each e3nn submodule before the outer
    Metatomic wrapper is scripted.

    Networks without MACE/e3nn components are returned unchanged.
    """
    network = network.eval()

    if not _contains_e3nn_or_mace(network):
        return network

    try:
        from e3nn.util.jit import script as e3nn_script
    except ImportError as exc:
        raise ImportError(
            "Exporting a MACE/e3nn model requires e3nn to be installed."
        ) from exc

    try:
        return e3nn_script(
            network,
            in_place=False,
        )
    except Exception as exc:
        raise RuntimeError(
            "Failed to prepare the MACE/e3nn network with "
            "e3nn.util.jit.script."
        ) from exc


def _make_inference_model(
    model: torch.nn.Module,
    interaction_range: float,
) -> CVInferenceModel:
    """Extract the inference pipeline stored in ``BaseCV.nn``."""
    model = model.eval()
    network = getattr(model, "nn", None)

    if network is None:
        raise ValueError(
            "The supplied mlcolvar model does not expose its inference "
            "network through the 'nn' attribute."
        )

    if getattr(model, "preprocessing", None) is not None:
        raise ValueError(
            "Metatomic export expects the complete raw-input-to-CV pipeline "
            "to be contained in model.nn. External BaseCV preprocessing is "
            "not supported by this exporter."
        )

    featurizer = getattr(network, "featurizer", None)

    if featurizer is None:
        raise ValueError(
            "Metatomic atomistic export requires model.nn.featurizer."
        )

    backbone = getattr(featurizer, "backbone", None)

    if backbone is None:
        raise ValueError(
            "Metatomic atomistic export requires "
            "model.nn.featurizer.backbone."
        )

    atomic_numbers = getattr(backbone, "atomic_numbers", None)

    if atomic_numbers is None:
        atomic_numbers = getattr(featurizer, "atomic_numbers", None)

    if atomic_numbers is None:
        raise ValueError(
            "The atomistic backbone or featurizer must expose "
            "'atomic_numbers'."
        )

    atomic_numbers = atomic_numbers.detach().to(dtype=torch.long)

    neighbor_options = _get_neighbor_options(
        network=network,
        interaction_range=interaction_range,
    )

    network = _prepare_network(network)

    postprocessing = getattr(model, "postprocessing", None)
    if postprocessing is None:
        postprocessing = torch.nn.Identity()
    postprocessing = postprocessing.eval()

    inference_model = CVInferenceModel(
        network=network,
        postprocessing=postprocessing,
        atomic_numbers=atomic_numbers,
        neighbor_options=neighbor_options,
    )
    inference_model.eval()

    for parameter in inference_model.parameters():
        parameter.requires_grad_(False)

    return inference_model


def create_metatomic_model(
    model: torch.nn.Module,
    out_features: int,
    atomic_types: Sequence[int],
    interaction_range: float,
    *,
    length_unit: str = "angstrom",
    dtype: str = "float64",
    supported_devices: Optional[Sequence[str]] = None,
    name: str = "mlcolvar collective variable",
    description: str = (
        "Collective variable combining a pretrained atomistic representation "
        "with an mlcolvar readout."
    ),
    authors: Optional[Sequence[str]] = None,
) -> MetatomicAtomisticModel:
    """Create an exportable Metatomic model from a trained mlcolvar CV."""
    if len(atomic_types) == 0:
        raise ValueError(
            "'atomic_types' must contain at least one atomic type."
        )

    if interaction_range < 0.0:
        raise ValueError(
            "'interaction_range' must be non-negative."
        )

    if dtype not in ("float32", "float64"):
        raise ValueError(
            "'dtype' must be either 'float32' or 'float64'."
        )

    if supported_devices is None:
        supported_devices = ("cpu", "cuda")

    if authors is None:
        authors = ()

    inference_model = _make_inference_model(
        model=model,
        interaction_range=interaction_range,
    )

    wrapper = MetatomicCVWrapper(
        model=inference_model,
        out_features=out_features,
    )
    wrapper.eval()

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
        atomic_types=[
            int(atomic_type)
            for atomic_type in atomic_types
        ],
        interaction_range=float(interaction_range),
        supported_devices=list(supported_devices),
        dtype=dtype,
    )

    return MetatomicAtomisticModel(
        module=wrapper,
        metadata=metadata,
        capabilities=capabilities,
    )


def export_metatomic_model(
    model: torch.nn.Module,
    path: Union[str, Path],
    out_features: int,
    atomic_types: Sequence[int],
    interaction_range: float,
    *,
    length_unit: str = "angstrom",
    dtype: str = "float64",
    supported_devices: Optional[Sequence[str]] = None,
    name: str = "mlcolvar collective variable",
    description: str = (
        "Collective variable combining a pretrained atomistic representation "
        "with an mlcolvar readout."
    ),
    authors: Optional[Sequence[str]] = None,
    collect_extensions: Optional[Union[str, Path]] = None,
) -> Path:
    """Create and save a trained mlcolvar CV in Metatomic format."""
    output_path = Path(path)

    if output_path.suffix != ".pt":
        raise ValueError(
            "The exported Metatomic model must use the '.pt' extension."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)

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
        metatomic_model.save(str(output_path))
    else:
        extension_path = Path(collect_extensions)
        extension_path.mkdir(parents=True, exist_ok=True)
        metatomic_model.save(
            str(output_path),
            collect_extensions=str(extension_path),
        )

    return output_path
