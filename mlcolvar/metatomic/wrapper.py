from typing import Dict, List, Optional

import torch

from ._compat import (
    Labels, ModelOutput, NeighborListOptions, System, TensorBlock, TensorMap,
)


class _CVInferenceModel(torch.nn.Module):
    """Build mlcolvar graph inputs from one or two Metatomic neighbor lists.

    In merged mode, the long-range list is filtered to the selected subsystem,
    appended after the short list, and flagged in ``edge_masks_lr``. If core
    system atoms are specified, environment membership is recomputed each step.
    """

    neighbor_options: NeighborListOptions
    neighbor_lists: List[NeighborListOptions]
    __constants__ = ["graph_mode", "dual_neighbors", "short_cutoff", "buffer", "use_long_for_environment"]

    def __init__(
        self,
        network: torch.nn.Module,
        postprocessing: torch.nn.Module,
        atomic_numbers: torch.Tensor,
        neighbor_options: NeighborListOptions,
        long_neighbor_options: Optional[NeighborListOptions] = None,
        short_cutoff: float = 0.0,
        long_cutoff: Optional[float] = None,
        graph_mode: str = "merged",
        system_indices: Optional[List[int]] = None,
        environment_indices: Optional[List[int]] = None,
        subsystem_indices: Optional[List[int]] = None,
        subsystem_atomic_numbers: Optional[List[int]] = None,
        buffer: float = 0.0,
    ) -> None:
        super().__init__()
        if graph_mode not in ("merged", "separate"):
            raise ValueError("`graph_mode` must be 'merged' or 'separate'.")
        self.dual_neighbors = long_neighbor_options is not None
        if graph_mode == "separate" and not self.dual_neighbors:
            raise ValueError("Separate graph mode requires a long neighbor list.")
        if buffer < 0:
            raise ValueError("`buffer` must be non-negative.")
        if short_cutoff <= 0:
            short_cutoff = float(neighbor_options.cutoff)
        for name, indices in (("system", system_indices),
                              ("environment", environment_indices),
                              ("subsystem", subsystem_indices)):
            if indices is not None and (any(i < 0 for i in indices) or len(indices) != len(set(indices))):
                raise ValueError(f"`{name}_indices` must be unique and non-negative.")
        if system_indices is not None and not system_indices:
            raise ValueError("`system_indices` must not be empty.")
        if environment_indices is not None and system_indices is None:
            raise ValueError("`environment_indices` requires `system_indices`.")
        if subsystem_indices is not None and subsystem_atomic_numbers is not None:
            raise ValueError("Select subsystem by indices OR atomic numbers.")
        if self.dual_neighbors and graph_mode == "merged" and subsystem_indices is None and subsystem_atomic_numbers is None:
            raise ValueError("Merged dual-cutoff graphs require a subsystem selection.")
        if graph_mode == "separate" and system_indices is not None:
            raise ValueError("Dynamic environment selection requires merged mode.")

        self.network = network
        self.postprocessing = postprocessing
        self.neighbor_options = neighbor_options
        self.neighbor_lists = [neighbor_options]
        if long_neighbor_options is not None:
            self.neighbor_lists.append(long_neighbor_options)
        self.short_cutoff = float(short_cutoff)
        self.buffer = float(buffer)
        self.use_long_for_environment = (long_neighbor_options is not None and
            long_cutoff is not None and buffer > 0 and long_cutoff >= short_cutoff + buffer)
        self.graph_mode = graph_mode
        self.use_system_selection = system_indices is not None
        self.use_environment_selection = environment_indices is not None
        self.use_subsystem_types = subsystem_atomic_numbers is not None
        self.register_buffer("atomic_numbers", atomic_numbers.to(dtype=torch.long))
        self.register_buffer("system_indices", torch.tensor(system_indices or [], dtype=torch.long))
        self.register_buffer("environment_indices", torch.tensor(environment_indices or [], dtype=torch.long))
        self.register_buffer("subsystem_indices", torch.tensor(subsystem_indices or [], dtype=torch.long))
        self.register_buffer("subsystem_atomic_numbers", torch.tensor(subsystem_atomic_numbers or [], dtype=torch.long))

    def requested_neighbor_lists(self) -> List[NeighborListOptions]:
        return self.neighbor_lists

    def _systems_to_graph(self, systems: List[System]) -> Dict[str, torch.Tensor]:
        positions_list = torch.jit.annotate(List[torch.Tensor], [])
        node_attrs_list = torch.jit.annotate(List[torch.Tensor], [])
        batch_list = torch.jit.annotate(List[torch.Tensor], [])
        cell_list = torch.jit.annotate(List[torch.Tensor], [])
        pbc_list = torch.jit.annotate(List[torch.Tensor], [])
        ptr_list = torch.jit.annotate(List[int], [0])
        short_edges = torch.jit.annotate(List[torch.Tensor], [])
        short_shifts = torch.jit.annotate(List[torch.Tensor], [])
        short_units = torch.jit.annotate(List[torch.Tensor], [])
        long_edges = torch.jit.annotate(List[torch.Tensor], [])
        long_shifts = torch.jit.annotate(List[torch.Tensor], [])
        long_units = torch.jit.annotate(List[torch.Tensor], [])
        system_masks = torch.jit.annotate(List[torch.Tensor], [])
        subsystem_masks = torch.jit.annotate(List[torch.Tensor], [])
        system_counts = torch.jit.annotate(List[int], [])
        atom_offset = 0

        for system_index, system in enumerate(systems):
            positions = system.positions
            device, dtype = positions.device, positions.dtype
            cell = system.cell.to(device=device, dtype=dtype)
            n_atoms = positions.shape[0]
            types = system.types.to(device=device, dtype=torch.long)
            positions_list.append(positions)
            node_attrs_list.append((types[:, None] == self.atomic_numbers[None, :]).to(dtype))
            batch_list.append(torch.full((n_atoms,), system_index, device=device, dtype=torch.long))
            cell_list.append(cell)
            pbc_list.append(system.pbc.to(device=device))

            core = torch.ones(n_atoms, dtype=torch.bool, device=device)
            if self.use_system_selection:
                core = torch.zeros(n_atoms, dtype=torch.bool, device=device)
                core_idx = self.system_indices.to(device)
                if torch.any(core_idx >= n_atoms):
                    raise ValueError("`system_indices` exceed the system atom count.")
                core[core_idx] = True
            system_masks.append(core.reshape(-1, 1))
            system_counts.append(int(core.sum()))

            subgroup = torch.zeros(n_atoms, dtype=torch.bool, device=device)
            if self.dual_neighbors and self.graph_mode == "merged":
                if self.use_subsystem_types:
                    for z in self.subsystem_atomic_numbers:
                        subgroup = subgroup | (types == z)
                else:
                    sub_idx = self.subsystem_indices.to(device)
                    if torch.any(sub_idx >= n_atoms):
                        raise ValueError("`subsystem_indices` exceed the system atom count.")
                    subgroup[sub_idx] = True
            subsystem_masks.append(subgroup.reshape(-1, 1))

            short_samples = system.get_neighbor_list(self.neighbor_options).samples.values
            long_samples = torch.empty((0, 5), dtype=torch.int32, device=device)
            if self.dual_neighbors:
                long_samples = system.get_neighbor_list(self.neighbor_lists[1]).samples.values
            src, dst = short_samples[:, 0].long(), short_samples[:, 1].long()
            shifts_short = short_samples[:, 2:5].to(device=device, dtype=dtype)
            distances = torch.linalg.vector_norm(
                positions[dst] - positions[src] + shifts_short @ cell, dim=1,
            )
            if self.use_system_selection:
                env = ~core
                if self.use_environment_selection:
                    env = torch.zeros(n_atoms, dtype=torch.bool, device=device)
                    env_idx = self.environment_indices.to(device)
                    if torch.any(env_idx >= n_atoms):
                        raise ValueError("`environment_indices` exceed the system atom count.")
                    env[env_idx] = True
                    if torch.any(env & core):
                        raise ValueError("System and environment selections must be disjoint.")
                candidate = long_samples if self.use_long_for_environment else short_samples
                candidate_src, candidate_dst = candidate[:, 0].long(), candidate[:, 1].long()
                candidate_shifts = candidate[:, 2:5].to(device=device, dtype=dtype)
                candidate_dist = torch.linalg.vector_norm(
                    positions[candidate_dst] - positions[candidate_src] + candidate_shifts @ cell, dim=1,
                )
                near_core = (candidate_dist <= self.short_cutoff + self.buffer) & (
                    (core[candidate_src] & env[candidate_dst]) |
                    (env[candidate_src] & core[candidate_dst])
                )
                active = core.clone()
                active.index_fill_(0, candidate_src[near_core], True)
                active.index_fill_(0, candidate_dst[near_core], True)
                keep_short = active[src] & active[dst] & (distances <= self.short_cutoff)
            else:
                keep_short = distances <= self.short_cutoff
            # Exclude unshifted self-edges to match mlcolvar's get_neighborhood.
            keep_short = keep_short & ((src != dst) | (shifts_short != 0).any(dim=1))
            short_samples = short_samples[keep_short]
            idx = short_samples[:, :2].to(device=device, dtype=torch.long).T + atom_offset
            units = short_samples[:, 2:5].to(device=device, dtype=dtype)
            short_edges.append(idx)
            short_units.append(units)
            short_shifts.append(units @ cell)

            if self.dual_neighbors:
                if self.graph_mode == "merged":
                    src_l, dst_l = long_samples[:, 0].long(), long_samples[:, 1].long()
                    keep_long = subgroup[src_l] & subgroup[dst_l]
                    # Never add an unshifted self edge.
                    keep_long = keep_long & ((src_l != dst_l) | (long_samples[:, 2:5] != 0).any(dim=1))
                    long_samples = long_samples[keep_long]
                long_idx = long_samples[:, :2].to(device=device, dtype=torch.long).T + atom_offset
                long_unit = long_samples[:, 2:5].to(device=device, dtype=dtype)
                long_edges.append(long_idx)
                long_units.append(long_unit)
                long_shifts.append(long_unit @ cell)

            atom_offset += n_atoms
            ptr_list.append(atom_offset)

        data = torch.jit.annotate(Dict[str, torch.Tensor], {})
        data["positions"] = torch.cat(positions_list, dim=0)
        data["node_attrs"] = torch.cat(node_attrs_list, dim=0)
        data["batch"] = torch.cat(batch_list, dim=0)
        data["ptr"] = torch.tensor(ptr_list, dtype=torch.long, device=systems[0].positions.device)
        data["cell"] = torch.stack(cell_list, dim=0)
        data["pbc"] = torch.stack(pbc_list, dim=0)
        short_idx, short_shift, short_unit = (
            torch.cat(short_edges, dim=1), torch.cat(short_shifts, dim=0), torch.cat(short_units, dim=0),
        )
        data["edge_index"], data["shifts"], data["unit_shifts"] = short_idx, short_shift, short_unit
        if self.use_system_selection:
            data["system_masks"] = torch.cat(system_masks, dim=0)
            data["n_system"] = torch.tensor(
                system_counts, dtype=data["positions"].dtype, device=data["positions"].device,
            ).reshape(-1, 1)
            data["n_env"] = (
                (data["ptr"][1:] - data["ptr"][:-1]).to(data["n_system"].dtype).reshape(-1, 1)
                - data["n_system"]
            )
        if self.dual_neighbors:
            long_idx, long_shift, long_unit = (
                torch.cat(long_edges, dim=1), torch.cat(long_shifts, dim=0), torch.cat(long_units, dim=0),
            )
            data["edge_index_short"], data["shifts_short"], data["unit_shifts_short"] = (
                short_idx, short_shift, short_unit,
            )
            data["edge_index_long"], data["shifts_long"], data["unit_shifts_long"] = (
                long_idx, long_shift, long_unit,
            )
            if self.graph_mode == "merged":
                data["edge_index"] = torch.cat((short_idx, long_idx), dim=1)
                data["shifts"] = torch.cat((short_shift, long_shift), dim=0)
                data["unit_shifts"] = torch.cat((short_unit, long_unit), dim=0)
                data["edge_masks_lr"] = torch.cat((
                    torch.zeros((short_idx.shape[1], 1), dtype=torch.bool, device=short_idx.device),
                    torch.ones((long_idx.shape[1], 1), dtype=torch.bool, device=long_idx.device),
                ), dim=0)
                data["subsystem_masks"] = torch.cat(subsystem_masks, dim=0)
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
