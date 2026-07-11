"""Metatomic export utilities for mlcolvar collective-variable models.

This module exports the inference model stored in ``BaseCV.nn`` through the
Metatomic interface. The original LightningModule is not retained, so
training-only state such as the Trainer, loss functions, optimizers, metrics,
and training hooks is excluded from the exported model.

The mlcolvar atomistic model expects a ``Dict[str, Tensor]`` graph batch,
whereas Metatomic provides ``List[System]``. ``CVInferenceModel`` converts
between these two interfaces before evaluating the trained model.
"""

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import torch

try:
    from metatensor.torch import Labels, TensorBlock, TensorMap
    from metatomic.torch import (
        AtomisticModel as MetatomicAtomisticModel,
        ModelCapabilities,
        ModelMetadata,
        ModelOutput,
        NeighborListOptions,
        System,
    )
except ImportError as exc:
    raise ImportError(
        "Metatomic export requires both 'metatensor-torch' and "
        "'metatomic-torch'. Install these optional dependencies before "
        "importing mlcolvar.integrations.metatomic."
    ) from exc


__all__ = [
    "CVInferenceModel",
    "MetatomicCVWrapper",
    "create_metatomic_model",
    "export_metatomic_model",
]


class CVInferenceModel(torch.nn.Module):
    """Adapt Metatomic systems to the mlcolvar atomistic-model interface.

    Parameters
    ----------
    network
        Complete mlcolvar atomistic inference model. It must accept a
        ``Dict[str, Tensor]`` containing graph data and return a tensor with
        shape ``(n_systems, out_features)``.
    postprocessing
        Optional BaseCV postprocessing module. ``Identity`` is used when no
        postprocessing is configured.
    atomic_numbers
        Atomic numbers supported by the pretrained backbone. They are used to
        build the one-hot ``node_attrs`` expected by mlcolvar atomistic models.
    neighbor_options
        Neighbor-list specification requested from the simulation engine.
    """

    neighbor_options: NeighborListOptions

    def __init__(
        self,
        network: torch.nn.Module,
        postprocessing: torch.nn.Module,
        atomic_numbers: torch.Tensor,
        neighbor_options: NeighborListOptions,
    ) -> None:
        super().__init__()

        self.network = network
        self.postprocessing = postprocessing
        self.neighbor_options = neighbor_options

        self.register_buffer(
            "atomic_numbers",
            atomic_numbers.to(dtype=torch.long),
        )

    def requested_neighbor_lists(
        self,
    ) -> List[NeighborListOptions]:
        """Return the neighbor list required by the wrapped model."""
        return [self.neighbor_options]

    def _systems_to_graph(
        self,
        systems: List[System],
    ) -> Dict[str, torch.Tensor]:
        """Convert Metatomic systems into the mlcolvar graph representation."""
        positions_list = torch.jit.annotate(
            List[torch.Tensor],
            [],
        )
        node_attrs_list = torch.jit.annotate(
            List[torch.Tensor],
            [],
        )
        edge_index_list = torch.jit.annotate(
            List[torch.Tensor],
            [],
        )
        unit_shifts_list = torch.jit.annotate(
            List[torch.Tensor],
            [],
        )
        batch_list = torch.jit.annotate(
            List[torch.Tensor],
            [],
        )
        cell_list = torch.jit.annotate(
            List[torch.Tensor],
            [],
        )
        pbc_list = torch.jit.annotate(
            List[torch.Tensor],
            [],
        )
        ptr_list = torch.jit.annotate(
            List[int],
            [0],
        )

        atom_offset = 0

        for system_index, system in enumerate(systems):
            positions = system.positions
            types = system.types.to(dtype=torch.long)
            n_atoms = positions.shape[0]

            positions_list.append(positions)
            cell_list.append(system.cell)
            pbc_list.append(system.pbc)

            node_attrs = (
                types.reshape(-1, 1)
                == self.atomic_numbers.reshape(1, -1)
            ).to(
                dtype=positions.dtype,
                device=positions.device,
            )
            node_attrs_list.append(node_attrs)

            batch_list.append(
                torch.full(
                    (n_atoms,),
                    system_index,
                    dtype=torch.long,
                    device=positions.device,
                )
            )

            neighbor_list = system.get_neighbor_list(
                self.neighbor_options
            )
            samples = neighbor_list.samples.values

            first_atom = samples[:, 0].to(
                dtype=torch.long,
                device=positions.device,
            )
            second_atom = samples[:, 1].to(
                dtype=torch.long,
                device=positions.device,
            )

            edge_index_list.append(
                torch.stack(
                    [
                        first_atom + atom_offset,
                        second_atom + atom_offset,
                    ],
                    dim=0,
                )
            )

            unit_shifts_list.append(
                samples[:, 2:5].to(
                    dtype=positions.dtype,
                    device=positions.device,
                )
            )

            atom_offset += n_atoms
            ptr_list.append(atom_offset)

        data = torch.jit.annotate(
            Dict[str, torch.Tensor],
            {},
        )

        data["positions"] = torch.cat(
            positions_list,
            dim=0,
        )
        data["node_attrs"] = torch.cat(
            node_attrs_list,
            dim=0,
        )
        data["edge_index"] = torch.cat(
            edge_index_list,
            dim=1,
        )
        data["unit_shifts"] = torch.cat(
            unit_shifts_list,
            dim=0,
        )
        data["batch"] = torch.cat(
            batch_list,
            dim=0,
        )
        data["ptr"] = torch.tensor(
            ptr_list,
            dtype=torch.long,
            device=systems[0].positions.device,
        )
        data["cell"] = torch.stack(
            cell_list,
            dim=0,
        )
        data["pbc"] = torch.stack(
            pbc_list,
            dim=0,
        )

        return data

    def forward(
        self,
        systems: List[System],
    ) -> torch.Tensor:
        """Evaluate the complete atomistic CV inference pipeline."""
        data = self._systems_to_graph(systems)
        features = self.network(data)
        return self.postprocessing(features)


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
            "e3nn.util.jit.script. Run the export in a fresh Python process "
            "and do not call torch.jit.script(model.nn) beforehand."
        ) from exc

def _make_inference_model(
    model: torch.nn.Module,
    interaction_range: float,
) -> CVInferenceModel:
    """Extract and adapt the inference path stored in ``BaseCV.nn``."""
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

    neighbor_options = _get_neighbor_options(
        network=network,
        interaction_range=interaction_range,
    )

    # MACE/e3nn modules must be prepared with e3nn's recursive JIT utility
    # before Metatomic scripts the outer wrapper. PET/DeepMD and other plain
    # TorchScript-compatible models are left unchanged.
    network = _prepare_network(network)

    postprocessing = getattr(model, "postprocessing", None)

    if postprocessing is None:
        postprocessing = torch.nn.Identity()

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


class MetatomicCVWrapper(torch.nn.Module):
    """Expose an inference-only mlcolvar CV through Metatomic.

    Parameters
    ----------
    model
        Pure ``torch.nn.Module`` accepting ``List[System]`` and returning a
        tensor with shape ``(n_systems, out_features)``.
    out_features
        Number of collective-variable components produced for each system.
    output_name
        Name of the Metatomic output. PLUMED expects ``"feature"``.
    property_name
        Name of the property labels indexing the CV components.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        out_features: int,
        output_name: str = "feature",
        property_name: str = "feature",
    ) -> None:
        super().__init__()

        if out_features < 1:
            raise ValueError(
                "'out_features' must be a positive integer."
            )

        if output_name != "feature":
            raise ValueError(
                "PLUMED requires the Metatomic output to be named 'feature'."
            )

        if property_name == "":
            raise ValueError("'property_name' must not be empty.")

        self.model = model
        self.out_features = int(out_features)
        self.output_name = output_name
        self.property_name = property_name

    def _keys(
        self,
        device: torch.device,
    ) -> Labels:
        """Create the single key used by the output TensorMap."""
        return Labels(
            names=["_"],
            values=torch.zeros(
                (1, 1),
                dtype=torch.int32,
                device=device,
            ),
        )

    def _properties(
        self,
        device: torch.device,
    ) -> Labels:
        """Create labels for the individual CV components."""
        return Labels(
            names=[self.property_name],
            values=torch.arange(
                self.out_features,
                dtype=torch.int32,
                device=device,
            ).reshape(-1, 1),
        )

    def _empty_feature(
        self,
        system: System,
    ) -> TensorMap:
        """Return the empty output used by PLUMED to infer the CV size."""
        device = system.positions.device
        dtype = system.positions.dtype

        components = torch.jit.annotate(List[Labels], [])

        block = TensorBlock(
            values=torch.zeros(
                (0, self.out_features),
                dtype=dtype,
                device=device,
            ),
            samples=Labels(
                names=["system"],
                values=torch.zeros(
                    (0, 1),
                    dtype=torch.int32,
                    device=device,
                ),
            ),
            components=components,
            properties=self._properties(device),
        )

        return TensorMap(
            keys=self._keys(device),
            blocks=[block],
        )

    def _features_to_tensor_map(
        self,
        features: torch.Tensor,
        n_systems: int,
        device: torch.device,
    ) -> TensorMap:
        """Convert dense system-level CV values to a TensorMap."""
        if features.ndim == 1:
            if n_systems == 1 and features.shape[0] == self.out_features:
                features = features.reshape(1, self.out_features)
            elif self.out_features == 1 and features.shape[0] == n_systems:
                features = features.reshape(n_systems, 1)
            else:
                raise ValueError(
                    "The one-dimensional model output is ambiguous."
                )

        if features.ndim != 2:
            raise ValueError(
                "The wrapped model must return a rank-two tensor."
            )

        if features.shape[0] != n_systems:
            raise ValueError(
                "The first output dimension must equal the number of systems."
            )

        if features.shape[1] != self.out_features:
            raise ValueError(
                "The second output dimension must equal 'out_features'."
            )

        if features.device != device:
            raise ValueError(
                "The model output and input systems must use the same device."
            )

        components = torch.jit.annotate(List[Labels], [])

        block = TensorBlock(
            values=features,
            samples=Labels(
                names=["system"],
                values=torch.arange(
                    n_systems,
                    dtype=torch.int32,
                    device=device,
                ).reshape(-1, 1),
            ),
            components=components,
            properties=self._properties(device),
        )

        return TensorMap(
            keys=self._keys(device),
            blocks=[block],
        )

    def forward(
        self,
        systems: List[System],
        outputs: Dict[str, ModelOutput],
        selected_atoms: Optional[Labels] = None,
    ) -> Dict[str, TensorMap]:
        """Evaluate the requested system-level collective variable."""
        result = torch.jit.annotate(Dict[str, TensorMap], {})

        if "feature" not in outputs:
            return result

        if outputs["feature"].sample_kind == "atom":
            raise ValueError(
                "MetatomicCVWrapper only supports system-level features."
            )

        if len(systems) == 0:
            raise ValueError("At least one System must be supplied.")

        if len(systems) == 1 and len(systems[0]) == 0:
            result["feature"] = self._empty_feature(systems[0])
            return result

        features = self.model(systems)

        result["feature"] = self._features_to_tensor_map(
            features=features,
            n_systems=len(systems),
            device=systems[0].positions.device,
        )

        return result


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
