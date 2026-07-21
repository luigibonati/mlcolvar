from typing import Dict, Optional

import torch


__all__ = [
    "infer_num_graphs",
    "get_graph_ptr",
    "prepare_cells",
    "prepare_pbc",
]


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
