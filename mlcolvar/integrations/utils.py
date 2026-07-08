from typing import Dict, Optional, Sequence, Union

import torch


__all__ = [
    "align_node_attrs",
    "get_graph_ptr",
    "infer_num_graphs",
    "prepare_cells",
    "prepare_pbc",
]


def _as_atomic_number_list(
    atomic_numbers: Union[Sequence[int], torch.Tensor],
    name: str,
) -> list[int]:
    numbers = (
        torch.as_tensor(atomic_numbers, dtype=torch.long)
        .detach()
        .cpu()
        .reshape(-1)
    )

    values = [int(number) for number in numbers.tolist()]

    if len(values) == 0:
        raise ValueError(f"The {name} atomic-number table cannot be empty.")

    if any(number <= 0 for number in values):
        raise ValueError(
            f"The {name} atomic-number table must contain positive integers."
        )

    if len(set(values)) != len(values):
        raise ValueError(
            f"The {name} atomic-number table contains duplicates: {values}."
        )

    return values


def align_node_attrs(
    dataset,
    target_atomic_numbers: Union[Sequence[int], torch.Tensor],
):
    """Align graph node attributes with an external model element table."""

    if not hasattr(dataset, "metadata"):
        raise TypeError("The dataset must expose a `metadata` attribute.")

    if "atomic_numbers" not in dataset.metadata:
        raise KeyError("The dataset metadata must contain `atomic_numbers`.")

    data_list = dataset["data_list"]

    source_values = _as_atomic_number_list(
        dataset.metadata["atomic_numbers"],
        name="source",
    )
    target_values = _as_atomic_number_list(
        target_atomic_numbers,
        name="target",
    )

    if source_values == target_values:
        return dataset

    target_indices = {
        atomic_number: index
        for index, atomic_number in enumerate(target_values)
    }

    missing = [
        atomic_number
        for atomic_number in source_values
        if atomic_number not in target_indices
    ]

    if missing:
        raise ValueError(
            "The target model does not support atomic numbers "
            f"{missing}."
        )

    source_to_target = torch.tensor(
        [target_indices[number] for number in source_values],
        dtype=torch.long,
    )

    for graph in data_list:
        old_node_attrs = graph["node_attrs"]

        if old_node_attrs.size(1) != len(source_values):
            raise ValueError(
                "`node_attrs` width does not match "
                "`dataset.metadata['atomic_numbers']`."
            )

        target_species = source_to_target.to(
            device=old_node_attrs.device,
        )[old_node_attrs.argmax(dim=-1)]

        new_node_attrs = old_node_attrs.new_zeros(
            old_node_attrs.size(0),
            len(target_values),
        )

        new_node_attrs.scatter_(
            dim=1,
            index=target_species.reshape(-1, 1),
            value=1,
        )

        graph["node_attrs"] = new_node_attrs

    dataset.metadata["atomic_numbers"] = target_values

    return dataset


def infer_num_graphs(
    data: Dict[str, torch.Tensor],
) -> int:
    """Infer the number of graphs in a graph batch."""

    if "ptr" in data:
        return int(data["ptr"].numel()) - 1

    if "n_system" in data:
        n_system = data["n_system"]
        return int(n_system.item()) if n_system.dim() == 0 else int(n_system.numel())

    if "batch" in data:
        batch = data["batch"]
        return 0 if batch.numel() == 0 else int(batch.max().item()) + 1

    raise RuntimeError(
        "Graph data must contain `ptr`, `n_system`, or `batch`."
    )


def get_graph_ptr(
    data: Dict[str, torch.Tensor],
    n_atoms: int,
    device: torch.device,
) -> torch.Tensor:
    """Return graph boundaries as a pointer tensor."""

    if "ptr" in data:
        return data["ptr"].to(
            device=device,
            dtype=torch.long,
        )

    if "batch" not in data:
        return torch.tensor(
            [0, n_atoms],
            device=device,
            dtype=torch.long,
        )

    batch = data["batch"].to(
        device=device,
        dtype=torch.long,
    )

    if batch.numel() == 0:
        return torch.zeros(
            1,
            device=device,
            dtype=torch.long,
        )

    counts = torch.bincount(
        batch,
        minlength=int(batch.max().item()) + 1,
    )

    return torch.cat(
        (
            torch.zeros(
                1,
                device=device,
                dtype=torch.long,
            ),
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

    cells = cells.to(
        device=positions.device,
        dtype=positions.dtype,
    )

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

    pbc = data["pbc"].to(
        device=cells.device,
        dtype=torch.bool,
    )

    if pbc.dim() == 1:
        if pbc.numel() == 3:
            pbc = pbc.reshape(1, 3).expand(n_systems, 3)
        elif pbc.numel() == 3 * n_systems:
            pbc = pbc.reshape(n_systems, 3)

    if pbc.dim() != 2 or pbc.size(0) != n_systems or pbc.size(1) != 3:
        raise ValueError(
            "Expected PBC shape [3] or [n_systems, 3]."
        )

    return pbc