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


class _CVInferenceModel(torch.nn.Module):
    """Adapt Metatomic systems to the mlcolvar graph interface."""

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
        self.register_buffer("atomic_numbers", atomic_numbers.to(dtype=torch.long))

    def requested_neighbor_lists(self) -> List[NeighborListOptions]:
        return [self.neighbor_options]

    def _systems_to_graph(self, systems: List[System]) -> Dict[str, torch.Tensor]:
        positions_list = torch.jit.annotate(List[torch.Tensor], [])
        node_attrs_list = torch.jit.annotate(List[torch.Tensor], [])
        edge_index_list = torch.jit.annotate(List[torch.Tensor], [])
        unit_shifts_list = torch.jit.annotate(List[torch.Tensor], [])
        shifts_list = torch.jit.annotate(List[torch.Tensor], [])
        batch_list = torch.jit.annotate(List[torch.Tensor], [])
        cell_list = torch.jit.annotate(List[torch.Tensor], [])
        pbc_list = torch.jit.annotate(List[torch.Tensor], [])
        ptr_list = torch.jit.annotate(List[int], [0])

        atom_offset = 0
        for system_index, system in enumerate(systems):
            positions = system.positions
            types = system.types.to(dtype=torch.long, device=positions.device)
            cell = system.cell.to(dtype=positions.dtype, device=positions.device)
            n_atoms = positions.shape[0]

            positions_list.append(positions)
            cell_list.append(cell)
            pbc_list.append(system.pbc.to(device=positions.device))
            node_attrs_list.append(
                (types[:, None] == self.atomic_numbers[None, :]).to(positions.dtype)
            )
            batch_list.append(
                torch.full((n_atoms,), system_index, dtype=torch.long, device=positions.device)
            )

            samples = system.get_neighbor_list(self.neighbor_options).samples.values
            edge_index_list.append(
                samples[:, :2].to(dtype=torch.long, device=positions.device).transpose(0, 1)
                + atom_offset
            )

            unit_shifts = samples[:, 2:5].to(dtype=positions.dtype, device=positions.device)
            unit_shifts_list.append(unit_shifts)
            shifts_list.append(unit_shifts @ cell)

            atom_offset += n_atoms
            ptr_list.append(atom_offset)

        data = torch.jit.annotate(Dict[str, torch.Tensor], {})
        data["positions"] = torch.cat(positions_list, dim=0)
        data["node_attrs"] = torch.cat(node_attrs_list, dim=0)
        data["edge_index"] = torch.cat(edge_index_list, dim=1)
        data["unit_shifts"] = torch.cat(unit_shifts_list, dim=0)
        data["shifts"] = torch.cat(shifts_list, dim=0)
        data["batch"] = torch.cat(batch_list, dim=0)
        data["ptr"] = torch.tensor(ptr_list, dtype=torch.long, device=systems[0].positions.device)
        data["cell"] = torch.stack(cell_list, dim=0)
        data["pbc"] = torch.stack(pbc_list, dim=0)
        return data

    def forward(self, systems: List[System]) -> torch.Tensor:
        return self.postprocessing(self.network(self._systems_to_graph(systems)))


class _MetatomicCVWrapper(torch.nn.Module):
    """Expose an inference-only mlcolvar model through Metatomic."""

    __constants__ = ["sample_kind", "out_features"]

    def __init__(
        self, model: torch.nn.Module, out_features: int, sample_kind: str = "system"
    ) -> None:
        super().__init__()
        if out_features < 1:
            raise ValueError("`out_features` must be positive.")
        if sample_kind not in ("system", "atom"):
            raise ValueError("`sample_kind` must be 'system' or 'atom'.")
        self.model = model
        self.out_features = int(out_features)
        self.sample_kind = sample_kind

    def _tensor_map(self, values: torch.Tensor, samples: torch.Tensor) -> TensorMap:
        device = values.device
        components = torch.jit.annotate(List[Labels], [])
        sample_names = ["system", "atom"] if self.sample_kind == "atom" else ["system"]
        block = TensorBlock(
            values=values,
            samples=Labels(names=sample_names, values=samples),
            components=components,
            properties=Labels(
                names=["feature"],
                values=torch.arange(self.out_features, dtype=torch.int32, device=device)
                .reshape(-1, 1),
            ),
        )
        return TensorMap(
            keys=Labels(names=["_"], values=torch.zeros((1, 1), dtype=torch.int32, device=device)),
            blocks=[block],
        )

    def _empty_feature(self, system: System) -> TensorMap:
        columns = 2 if self.sample_kind == "atom" else 1
        return self._tensor_map(
            torch.zeros(
                (0, self.out_features), dtype=system.positions.dtype,
                device=system.positions.device,
            ),
            torch.zeros((0, columns), dtype=torch.int32, device=system.positions.device),
        )

    def _features_to_tensor_map(
        self,
        features: torch.Tensor,
        systems: List[System],
        selected_atoms: Optional[Labels],
    ) -> TensorMap:
        device = systems[0].positions.device
        n_systems = len(systems)
        offsets = torch.jit.annotate(List[int], [])
        counts = torch.jit.annotate(List[int], [])
        n_atoms = 0
        for system in systems:
            offsets.append(n_atoms)
            counts.append(len(system))
            n_atoms += len(system)

        expected_rows = n_atoms if self.sample_kind == "atom" else n_systems
        if features.ndim == 1:
            if expected_rows == 1 and features.shape[0] == self.out_features:
                features = features.reshape(1, self.out_features)
            elif self.out_features == 1 and features.shape[0] == expected_rows:
                features = features.reshape(expected_rows, 1)
            else:
                raise ValueError("The one-dimensional model output is ambiguous.")

        if features.ndim != 2:
            raise ValueError("The wrapped model must return a rank-two tensor.")
        if features.shape[0] != expected_rows:
            raise ValueError("Output rows must match the requested sample kind.")
        if features.shape[1] != self.out_features:
            raise ValueError("The second output dimension must equal `out_features`.")
        if features.device != device:
            raise ValueError("The model output and input systems must use the same device.")

        if self.sample_kind == "system":
            samples = torch.arange(n_systems, dtype=torch.int32, device=device).reshape(-1, 1)
        elif selected_atoms is None:
            sample_list = torch.jit.annotate(List[torch.Tensor], [])
            for i, system in enumerate(systems):
                atoms = torch.arange(len(system), dtype=torch.int32, device=device)
                system_ids = torch.full_like(atoms, i)
                sample_list.append(torch.stack((system_ids, atoms), dim=1))
            samples = torch.cat(sample_list, dim=0)
        else:
            selection = selected_atoms.values.to(device=device, dtype=torch.long)
            if selection.ndim != 2 or selection.shape[1] != 2:
                raise ValueError("`selected_atoms` must contain [system, atom] indices.")
            system_ids, atom_ids = selection[:, 0], selection[:, 1]
            if torch.any(system_ids < 0) or torch.any(system_ids >= n_systems):
                raise ValueError("Selected system index out of range.")
            atom_counts = torch.tensor(counts, dtype=torch.long, device=device)
            if torch.any(atom_ids < 0) or torch.any(atom_ids >= atom_counts[system_ids]):
                raise ValueError("Selected atom index out of range.")
            atom_offsets = torch.tensor(offsets, dtype=torch.long, device=device)
            features = features.index_select(0, atom_offsets[system_ids] + atom_ids)
            samples = selection.to(dtype=torch.int32)

        return self._tensor_map(features, samples)

    def forward(
        self,
        systems: List[System],
        outputs: Dict[str, ModelOutput],
        selected_atoms: Optional[Labels] = None,
    ) -> Dict[str, TensorMap]:
        result = torch.jit.annotate(Dict[str, TensorMap], {})
        if "feature" not in outputs:
            return result
        if outputs["feature"].sample_kind != self.sample_kind:
            raise ValueError("Requested sample kind does not match the exported model.")
        if len(systems) == 0:
            raise ValueError("At least one System must be supplied.")
        if len(systems) == 1 and len(systems[0]) == 0:
            result["feature"] = self._empty_feature(systems[0])
            return result

        features = self.model(systems)
        result["feature"] = self._features_to_tensor_map(features, systems, selected_atoms)
        return result
