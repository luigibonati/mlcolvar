from typing import Optional

import torch
from torch import nn
from torch_geometric.loader import DataLoader as GraphDataLoader

from mlcolvar.data import DictDataset

from ..base import Representation
from ._utils import require_graph_dataset, resolve_devices, temporary_eval

__all__ = [
    "compute_jacobian",
    "JacobianTransform",
]


def _indices(
    n: int,
    indices: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Normalize selected sample indices."""
    if indices is None:
        return torch.arange(n, dtype=torch.long)

    indices = torch.as_tensor(indices).cpu()

    if indices.dtype == torch.bool:
        indices = indices.reshape(-1)
        if len(indices) != n:
            raise ValueError(
                "Boolean `indices` must have the same length as the dataset."
            )
        indices = torch.nonzero(indices, as_tuple=False).reshape(-1)
    else:
        indices = indices.reshape(-1).long()

    indices = torch.unique(indices, sorted=True)

    if torch.any(indices < 0) or torch.any(indices >= n):
        raise IndexError("`indices` contains out-of-range indices.")

    return indices


def _compute_vector_jacobian(
    representation: Representation,
    dataset: DictDataset,
    indices: torch.Tensor,
    batch_size: int,
    device: torch.device,
    output_device: torch.device,
) -> torch.Tensor:
    """Compute Jacobians for vector representation inputs."""
    if "data" not in dataset.keys:
        raise KeyError("Vector Jacobian computation requires `data`.")

    x = dataset["data"]
    cell = dataset["cell"] if "cell" in dataset.keys else None
    jacobians = []

    with temporary_eval(representation, device), torch.enable_grad():
        for start in range(0, len(indices), batch_size):
            idx = indices[start : start + batch_size]

            xb = (
                x[idx.to(x.device)]
                .to(device)
                .detach()
                .requires_grad_(True)
            )
            cb = (
                None
                if cell is None
                else cell[idx.to(cell.device)].to(device)
            )

            output = representation(xb, cell=cb).reshape(len(xb), -1)

            gradients = [
                torch.autograd.grad(
                    output[:, i].sum(),
                    xb,
                    retain_graph=i + 1 < output.shape[-1],
                )[0]
                for i in range(output.shape[-1])
            ]

            jacobians.append(
                torch.stack(gradients, dim=-1)
                .detach()
                .to(output_device)
            )

    return torch.cat(jacobians)


def _compute_graph_jacobian(
    representation: Representation,
    dataset: DictDataset,
    indices: torch.Tensor,
    batch_size: int,
    device: torch.device,
    output_device: torch.device,
) -> torch.Tensor:
    """Compute Jacobians with respect to graph positions."""
    require_graph_dataset(dataset)

    selected_graphs = [
        dataset["data_list"][int(index)]
        for index in indices
    ]
    loader = GraphDataLoader(
        selected_graphs,
        batch_size=batch_size,
        shuffle=False,
    )

    jacobians = []
    n_atoms_ref = None

    with temporary_eval(representation, device):
        for graph in loader:
            graph = graph.to(device)
            n_graphs = int(graph.num_graphs)

            counts = torch.bincount(
                graph.batch,
                minlength=n_graphs,
            )

            if not torch.all(counts == counts[0]):
                raise ValueError(
                    "Selected graphs must have equal atom counts."
                )

            n_atoms = int(counts[0])
            if n_atoms_ref is None:
                n_atoms_ref = n_atoms
            elif n_atoms != n_atoms_ref:
                raise ValueError(
                    "Selected graphs must have equal atom counts."
                )

            graph.positions = (
                graph.positions
                .detach()
                .requires_grad_(True)
            )

            with torch.enable_grad():
                output = representation(graph)

                if output.ndim != 2 or output.shape[0] != n_graphs:
                    raise ValueError(
                        "Representation outputs must contain "
                        "one row per graph."
                    )

                gradients = []
                for i in range(output.shape[-1]):
                    gradient = torch.autograd.grad(
                        output[:, i].sum(),
                        graph.positions,
                        retain_graph=i + 1 < output.shape[-1],
                    )[0]

                    gradients.append(
                        torch.stack([
                            gradient[graph.batch == j]
                            for j in range(n_graphs)
                        ])
                    )

                jacobians.append(
                    torch.stack(gradients, dim=-1)
                    .detach()
                    .to(output_device)
                )

    return torch.cat(jacobians)


def compute_jacobian(
    representation: Representation,
    dataset: DictDataset,
    *,
    indices: Optional[torch.Tensor] = None,
    batch_size: Optional[int] = None,
    device=None,
    output_device="cpu",
) -> torch.Tensor:
    """Compute Jacobians of a frozen representation with respect to its inputs."""
    if not isinstance(representation, Representation):
        raise TypeError(
            "`representation` must derive from `Representation`."
        )

    if not representation.freeze:
        raise RuntimeError(
            "Jacobian computation requires a frozen representation."
        )

    device, output_device = resolve_devices(
        representation,
        device,
        output_device,
    )

    selected = _indices(len(dataset), indices)
    if selected.numel() == 0:
        raise ValueError(
            "No samples selected for Jacobian computation."
        )

    if representation.input_kind == "vector":
        return _compute_vector_jacobian(
            representation,
            dataset,
            selected,
            batch_size or 1024,
            device,
            output_device,
        )

    if representation.input_kind == "graph":
        if representation.output_kind != "system":
            raise ValueError(
                "Graph Jacobians require system-level "
                "representation outputs. Use pooling or "
                "`concat_atoms()` first."
            )

        return _compute_graph_jacobian(
            representation,
            dataset,
            selected,
            batch_size or 256,
            device,
            output_device,
        )

    raise ValueError(
        "Unsupported representation input kind: "
        f"{representation.input_kind!r}."
    )


class JacobianTransform(nn.Module):
    """Apply a materialized Jacobian through the chain rule."""

    def __init__(self, jacobian: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer(
            "jacobian",
            jacobian,
            persistent=False,
        )

    def forward(
        self,
        gradient_output: torch.Tensor,
        ref_idx: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if ref_idx is None:
            raise ValueError("`ref_idx` is required.")

        ref_idx = ref_idx.reshape(-1).to(
            self.jacobian.device,
            dtype=torch.long,
        )

        if torch.any(ref_idx < 0) or torch.any(
            ref_idx >= len(self.jacobian)
        ):
            raise IndexError(
                "Invalid Jacobian index."
            )

        jacobian = self.jacobian[ref_idx].to(
            device=gradient_output.device,
            dtype=gradient_output.dtype,
        )

        return torch.einsum(
            "bl,b...l->b...",
            gradient_output,
            jacobian,
        )