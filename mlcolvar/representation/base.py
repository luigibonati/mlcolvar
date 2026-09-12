from typing import Any, Dict, Optional, Sequence

import torch
from torch import nn


__all__ = [
    "Representation",
    "TensorRepresentation",
    "GraphRepresentation",
    "as_positive_int",
    "module_reference_tensor",
    "infer_num_graphs",
    "get_graph_ptr",
    "prepare_cells",
    "prepare_pbc",
    "align_node_attrs",
]


def as_positive_int(value: Any, name: str) -> int:
    """Convert a scalar value to a positive Python integer."""
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError(f"`{name}` must be scalar.")
        value = value.detach().cpu().item()

    value = int(value)
    if value <= 0:
        raise ValueError(f"`{name}` must be positive. Found {value}.")
    return value


def module_reference_tensor(module: nn.Module) -> torch.Tensor:
    """Return a scalar tensor matching a module floating dtype/device."""
    for tensor in module.parameters():
        if tensor.is_floating_point() or tensor.is_complex():
            return torch.empty((), dtype=tensor.dtype, device=tensor.device)

    for tensor in module.buffers():
        if tensor.is_floating_point() or tensor.is_complex():
            return torch.empty((), dtype=tensor.dtype, device=tensor.device)

    return torch.empty(())


def infer_num_graphs(data: Dict[str, torch.Tensor]) -> int:
    """Infer the number of systems represented by a graph dictionary."""
    if "ptr" in data:
        return int(data["ptr"].numel()) - 1

    if "n_system" in data:
        n_system = data["n_system"]
        return int(n_system.item()) if n_system.dim() == 0 else int(n_system.numel())

    if "batch" in data:
        batch = data["batch"]
        return 0 if batch.numel() == 0 else int(batch.max().item()) + 1

    raise RuntimeError("Graph data must contain `ptr`, `n_system`, or `batch`.")


def get_graph_ptr(
    data: Dict[str, torch.Tensor],
    n_atoms: int,
    device: torch.device,
) -> torch.Tensor:
    """Return graph boundaries as a pointer tensor."""
    if "ptr" in data:
        return data["ptr"].to(device=device, dtype=torch.long)

    if "batch" not in data:
        return torch.tensor([0, n_atoms], device=device, dtype=torch.long)

    batch = data["batch"].to(device=device, dtype=torch.long)
    if batch.numel() == 0:
        return torch.zeros(1, device=device, dtype=torch.long)

    counts = torch.bincount(batch, minlength=int(batch.max().item()) + 1)
    return torch.cat(
        (
            torch.zeros(1, device=device, dtype=torch.long),
            counts.cumsum(dim=0),
        ),
        dim=0,
    )


def prepare_cells(
    data: Dict[str, torch.Tensor],
    cell: Optional[torch.Tensor],
    n_systems: int,
    positions: torch.Tensor,
) -> torch.Tensor:
    """Normalize cells to shape ``[n_systems, 3, 3]``."""
    if cell is not None:
        cells = cell
    elif "cell" in data:
        cells = data["cell"]
    else:
        return positions.new_zeros((n_systems, 3, 3))

    cells = cells.to(device=positions.device, dtype=positions.dtype)

    if cells.dim() == 2:
        if cells.size(0) == 3 and cells.size(1) == 3:
            cells = cells.unsqueeze(0)
        elif cells.size(0) == 3 * n_systems and cells.size(1) == 3:
            cells = cells.reshape(n_systems, 3, 3)

    if (
        cells.dim() != 3
        or cells.size(0) != n_systems
        or cells.size(1) != 3
        or cells.size(2) != 3
    ):
        raise ValueError(
            "Expected cell shape [3, 3], [n_systems, 3, 3], "
            "or [3 * n_systems, 3]."
        )

    return cells


def prepare_pbc(
    data: Dict[str, torch.Tensor],
    cells: torch.Tensor,
    n_systems: int,
) -> torch.Tensor:
    """Return PBC flags with shape ``[n_systems, 3]``."""
    if "pbc" not in data:
        return torch.linalg.vector_norm(cells, dim=-1) > 0.0

    pbc = data["pbc"].to(device=cells.device, dtype=torch.bool)

    if pbc.dim() == 1:
        if pbc.numel() == 3:
            pbc = pbc.reshape(1, 3).expand(n_systems, 3)
        elif pbc.numel() == 3 * n_systems:
            pbc = pbc.reshape(n_systems, 3)

    if pbc.dim() != 2 or pbc.size(0) != n_systems or pbc.size(1) != 3:
        raise ValueError("Expected PBC shape [3] or [n_systems, 3].")

    return pbc


def _as_atomic_number_list(
    atomic_numbers: Sequence[int] | torch.Tensor,
    name: str,
) -> list[int]:
    numbers = (
        torch.as_tensor(atomic_numbers, dtype=torch.long)
        .detach()
        .cpu()
        .reshape(-1)
    )
    values = [int(number) for number in numbers.tolist()]

    if not values:
        raise ValueError(f"The {name} atomic-number table cannot be empty.")
    if any(number <= 0 for number in values):
        raise ValueError(
            f"The {name} atomic-number table must contain positive integers."
        )
    if len(set(values)) != len(values):
        raise ValueError(f"The {name} atomic-number table contains duplicates: {values}.")

    return values


def align_node_attrs(
    dataset,
    target_atomic_numbers: Sequence[int] | torch.Tensor,
):
    """Align graph one-hot node attributes with a representation element table."""
    if not hasattr(dataset, "metadata"):
        raise TypeError("The dataset must expose a `metadata` attribute.")
    if "atomic_numbers" not in dataset.metadata:
        raise KeyError("The dataset metadata must contain `atomic_numbers`.")

    data_list = dataset["data_list"]
    source_values = _as_atomic_number_list(
        dataset.metadata["atomic_numbers"], name="source"
    )
    target_values = _as_atomic_number_list(target_atomic_numbers, name="target")

    if source_values == target_values:
        return dataset

    target_indices = {
        atomic_number: index for index, atomic_number in enumerate(target_values)
    }
    missing = [
        atomic_number
        for atomic_number in source_values
        if atomic_number not in target_indices
    ]
    if missing:
        raise ValueError(
            "The target representation does not support atomic numbers "
            f"{missing}."
        )

    source_to_target = torch.tensor(
        [target_indices[number] for number in source_values], dtype=torch.long
    )

    for graph in data_list:
        old_node_attrs = graph["node_attrs"]
        if old_node_attrs.size(1) != len(source_values):
            raise ValueError(
                "`node_attrs` width does not match "
                "`dataset.metadata['atomic_numbers']`."
            )

        target_species = source_to_target.to(device=old_node_attrs.device)[
            old_node_attrs.argmax(dim=-1)
        ]
        new_node_attrs = old_node_attrs.new_zeros(
            old_node_attrs.size(0), len(target_values)
        )
        new_node_attrs.scatter_(
            dim=1,
            index=target_species.reshape(-1, 1),
            value=1,
        )
        graph["node_attrs"] = new_node_attrs

    dataset.metadata["atomic_numbers"] = target_values
    return dataset


class Representation(nn.Module):
    """Base class for reusable frozen or trainable representations.

    A representation maps raw model input to a latent tensor.  It deliberately
    contains no task-specific readout.
    """

    __constants__ = [
        "input_kind",
        "output_kind",
        "sample_kind",
        "out_features",
        "freeze",
    ]

    def __init__(
        self,
        *,
        out_features: int,
        input_kind: str,
        output_kind: str,
        freeze: bool = True,
    ) -> None:
        super().__init__()

        if input_kind not in {"tensor", "graph"}:
            raise ValueError("`input_kind` must be 'tensor' or 'graph'.")
        if output_kind not in {"atom", "system"}:
            raise ValueError("`output_kind` must be 'atom' or 'system'.")

        self.out_features = as_positive_int(out_features, "out_features")
        self.input_kind = input_kind
        self.output_kind = output_kind
        # Temporary alias for code/tests written against the old atomistic API.
        self.sample_kind = output_kind
        self.freeze = bool(freeze)

    def _freeze_module(self, module: nn.Module) -> None:
        if not self.freeze:
            return

        if not isinstance(module, torch.jit.ScriptModule):
            module.requires_grad_(False)

        module.eval()

    def train(self, mode: bool = True):
        # A frozen representation is an inference module even when the
        # downstream task head is switched to train mode. Keeping the adapter
        # itself in eval mode matters for backbones such as MACE that inspect
        # ``self.training`` inside ``forward``.
        if self.freeze:
            super().train(False)
            self._keep_frozen_modules_in_eval()
            return self

        super().train(mode)
        return self

    def _keep_frozen_modules_in_eval(self) -> None:
        """Keep registered pretrained children in eval mode when frozen."""
        for child in self.children():
            child.eval()


class TensorRepresentation(Representation):
    """Base representation accepting dense tensor inputs."""

    __constants__ = ["in_features"]

    def __init__(
        self,
        *,
        in_features: int,
        out_features: int,
        output_kind: str = "system",
        freeze: bool = True,
    ) -> None:
        super().__init__(
            out_features=out_features,
            input_kind="tensor",
            output_kind=output_kind,
            freeze=freeze,
        )
        self.in_features = as_positive_int(in_features, "in_features")


class GraphRepresentation(Representation):
    """Base representation accepting mlcolvar graph dictionaries."""

    __constants__ = ["full_neighbor_list"]

    def __init__(
        self,
        *,
        out_features: int,
        atomic_numbers: Sequence[int],
        cutoff: float,
        output_kind: str = "atom",
        buffer: float = 0.0,
        long_range_cutoff: float = -1.0,
        full_neighbor_list: bool = True,
        freeze: bool = True,
    ) -> None:
        atomic_numbers = [int(number) for number in atomic_numbers]

        if not atomic_numbers:
            raise ValueError("`atomic_numbers` cannot be empty.")
        if len(set(atomic_numbers)) != len(atomic_numbers):
            raise ValueError("`atomic_numbers` must not contain duplicates.")
        if any(number <= 0 for number in atomic_numbers):
            raise ValueError("`atomic_numbers` must contain positive integers.")
        if cutoff <= 0.0:
            raise ValueError("`cutoff` must be positive.")
        if buffer < 0.0:
            raise ValueError("`buffer` must be non-negative.")
        if long_range_cutoff >= 0.0 and long_range_cutoff <= cutoff:
            raise ValueError(
                "`long_range_cutoff` must be negative or larger than `cutoff`."
            )

        super().__init__(
            out_features=out_features,
            input_kind="graph",
            output_kind=output_kind,
            freeze=freeze,
        )

        # BaseCV uses in_features=None to identify graph models.
        self.in_features = None
        self.full_neighbor_list = bool(full_neighbor_list)

        self.register_buffer(
            "feature_dim",
            torch.tensor(self.out_features, dtype=torch.int64),
        )
        self.register_buffer(
            "atomic_numbers",
            torch.tensor(atomic_numbers, dtype=torch.int64),
        )
        self.register_buffer(
            "cutoff",
            torch.tensor(cutoff, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "buffer",
            torch.tensor(buffer, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "long_range_cutoff",
            torch.tensor(long_range_cutoff, dtype=torch.get_default_dtype()),
        )

    @torch.jit.unused
    def align_dataset(self, dataset):
        return align_node_attrs(dataset, self.atomic_numbers)
