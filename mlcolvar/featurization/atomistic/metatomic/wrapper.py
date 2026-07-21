from typing import Dict, List, Optional

import torch

from ._compat import (
    Labels,
    ModelOutput,
    NeighborListOptions,
    System,
    TensorBlock,
    TensorMap,
)


__all__ = [
    "CVInferenceModel",
    "MetatomicCVWrapper",
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
        shifts_list = torch.jit.annotate(
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
            types = system.types.to(
                dtype=torch.long,
                device=positions.device,
            )
            cell = system.cell.to(
                dtype=positions.dtype,
                device=positions.device,
            )
            pbc = system.pbc.to(device=positions.device)
            n_atoms = positions.shape[0]

            positions_list.append(positions)
            cell_list.append(cell)
            pbc_list.append(pbc)

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

            unit_shifts = samples[:, 2:5].to(
                dtype=positions.dtype,
                device=positions.device,
            )
            unit_shifts_list.append(unit_shifts)

            # MACE requires Cartesian periodic-image shifts in addition to
            # the integer cell offsets supplied by Metatomic.
            shifts_list.append(
                torch.matmul(unit_shifts, cell)
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
        data["shifts"] = torch.cat(
            shifts_list,
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
