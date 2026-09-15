import copy
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
    """Convert a scalar or scalar tensor to float."""
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().item())
    return float(value)


def _get_neighbor_options(
    network: torch.nn.Module,
    interaction_range: float,
) -> NeighborListOptions:
    """Resolve the neighbor-list options required by the representation."""

    representation = getattr(network, "representation", None)
    candidates = [representation, network]

    for module in candidates:
        if module is None:
            continue

        request = getattr(module, "requested_neighbor_lists", None)
        if request is not None:
            options = request()

            if len(options) == 1:
                return options[0]

            if len(options) > 1:
                raise ValueError(
                    "The representation requests multiple neighbor lists; "
                    "only one is currently supported."
                )

    for module in candidates:
        if module is None:
            continue

        options = getattr(module, "neighbor_options", None)
        if options is not None:
            return options

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


def _prepare_network(
    network: torch.nn.Module,
) -> torch.nn.Module:
    """Prepare backend-specific modules for TorchScript export."""

    network = copy.deepcopy(network).eval()

    representation = getattr(network, "representation", None)
    if representation is None:
        return network

    prepare = getattr(
        representation,
        "prepare_for_torchscript",
        None,
    )

    if prepare is not None:
        try:
            prepare()
        except Exception as exc:
            raise RuntimeError(
                "Failed to prepare the representation "
                "for TorchScript export."
            ) from exc

    return network


def _make_inference_model(
    model: torch.nn.Module,
    interaction_range: float,
) -> CVInferenceModel:
    """Build the inference-only model used for Metatomic export."""

    model = model.eval()
    network = getattr(model, "nn", None)

    if network is None:
        raise ValueError(
            "The supplied model does not expose its network through `nn`."
        )

    if getattr(model, "preprocessing", None) is not None:
        raise ValueError(
            "Metatomic export requires preprocessing to be contained "
            "inside model.nn."
        )

    representation = getattr(network, "representation", None)
    if representation is None:
        raise ValueError(
            "Metatomic export requires model.nn.representation."
        )

    atomic_numbers = getattr(
        representation,
        "atomic_numbers",
        None,
    )

    if atomic_numbers is None:
        raise ValueError(
            "The representation must expose `atomic_numbers`."
        )

    atomic_numbers = (
        atomic_numbers.detach().cpu().to(torch.long)
    )

    neighbor_options = _get_neighbor_options(
        network,
        interaction_range,
    )

    network = _prepare_network(network)

    postprocessing = getattr(
        model,
        "postprocessing",
        None,
    )

    if postprocessing is None:
        postprocessing = torch.nn.Identity()

    inference_model = CVInferenceModel(
        network=network,
        postprocessing=postprocessing.eval(),
        atomic_numbers=atomic_numbers,
        neighbor_options=neighbor_options,
    ).eval()

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
        "Collective variable combining a pretrained atomistic "
        "representation with an mlcolvar readout."
    ),
    authors: Optional[Sequence[str]] = None,
) -> MetatomicAtomisticModel:
    """Create an exportable Metatomic model."""

    if not atomic_types:
        raise ValueError(
            "`atomic_types` must contain at least one atomic type."
        )

    if interaction_range < 0:
        raise ValueError(
            "`interaction_range` must be non-negative."
        )

    if dtype not in ("float32", "float64"):
        raise ValueError(
            "`dtype` must be 'float32' or 'float64'."
        )

    supported_devices = (
        ("cpu", "cuda")
        if supported_devices is None
        else supported_devices
    )

    authors = () if authors is None else authors

    inference_model = _make_inference_model(
        model,
        interaction_range,
    )

    wrapper = MetatomicCVWrapper(
        model=inference_model,
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
        atomic_types=[
            int(value)
            for value in atomic_types
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
        "Collective variable combining a pretrained atomistic "
        "representation with an mlcolvar readout."
    ),
    authors: Optional[Sequence[str]] = None,
    collect_extensions: Optional[Union[str, Path]] = None,
) -> Path:
    """Create and save a Metatomic model."""

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