# TODO: Move this module to `mlcolvar.utils.derivatives`, as the Jacobian utilities are generic and not specific to representations?
import torch
from torch import nn
from torch_geometric.loader import DataLoader as GraphDataLoader

from mlcolvar.data import DictDataset, DictLoader

from .._utils import module_reference_tensor

__all__ = ["compute_jacobian", "JacobianTransform"]


def _indices(
    n: int,
    indices: torch.Tensor | None = None,
) -> torch.Tensor:
    """Normalize and validate dataset indices."""
    if indices is None:
        return torch.arange(n)

    indices = torch.as_tensor(indices).cpu()
    if indices.dtype == torch.bool:
        if indices.numel() != n:
            raise ValueError(
                "Boolean `indices` must have the same length as the dataset."
            )
        indices = torch.where(indices)[0]
    else:
        indices = indices.flatten().long()

    indices = torch.unique(indices, sorted=True)
    if torch.any((indices < 0) | (indices >= n)):
        raise IndexError("`indices` contains out-of-range indices.")
    return indices


def compute_jacobian(
    model: nn.Module,
    dataset: DictDataset,
    *,
    indices: torch.Tensor | None = None,
    batch_size: int | None = None,
    device: torch.device | str | None = None,
    output_device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Compute model-output Jacobians with respect to dataset inputs."""
    selected = _indices(len(dataset), indices)
    if selected.numel() == 0:
        raise ValueError("No samples selected for Jacobian computation.")
    dataset = dataset[selected]

    reference = module_reference_tensor(model)
    original_device = reference.device
    training = model.training
    device = original_device if device is None else torch.device(device)
    output_device = torch.device(output_device)

    is_graph = dataset.metadata.get("data_type") == "graphs"
    if batch_size is None:
        batch_size = 256 if is_graph else 1024
    loader = (
        GraphDataLoader(dataset, batch_size=batch_size, shuffle=False)
        if is_graph
        else DictLoader(dataset, batch_size=batch_size, shuffle=False)
    )

    jacobians: list[torch.Tensor] = []
    n_atoms_ref: int | None = None
    model.to(device).eval()

    try:
        with torch.enable_grad():
            for batch in loader:
                if is_graph:
                    data = batch["data_list"].to(device)
                    counts = data.ptr[1:] - data.ptr[:-1]
                    if not torch.all(counts == counts[0]):
                        raise ValueError("Graphs must have equal atom counts.")

                    n_samples = len(counts)
                    n_atoms = int(counts[0])
                    if n_atoms_ref is None:
                        n_atoms_ref = n_atoms
                    elif n_atoms != n_atoms_ref:
                        raise ValueError("Graphs must have equal atom counts.")

                    data.positions = data.positions.detach().requires_grad_(True)
                    inputs = data.positions
                    output = model(data)
                else:
                    data = batch["data"].to(device).detach().requires_grad_(True)
                    cell = batch.get("cell")
                    cell = None if cell is None else cell.to(device)
                    n_samples = len(data)
                    inputs = data
                    output = (
                        model(data)
                        if cell is None
                        else model(data, cell=cell)
                    )

                output = output.reshape(n_samples, -1)
                gradients: list[torch.Tensor] = []
                for i in range(output.shape[-1]):
                    gradient = torch.autograd.grad(
                        output[:, i].sum(),
                        inputs,
                        retain_graph=i + 1 < output.shape[-1],
                    )[0]
                    if is_graph:
                        gradient = gradient.reshape(
                            n_samples,
                            n_atoms,
                            *gradient.shape[1:],
                        )
                    gradients.append(gradient)

                jacobians.append(
                    torch.stack(gradients, dim=-1)
                    .detach()
                    .to(output_device)
                )
    finally:
        model.to(original_device).train(training)

    return torch.cat(jacobians)


class JacobianTransform(nn.Module):
    """Propagate feature gradients through precomputed Jacobians."""

    def __init__(self, jacobian: torch.Tensor):
        super().__init__()
        self.register_buffer("jacobian", jacobian, persistent=False)

    def forward(
        self,
        gradient: torch.Tensor,
        ref_idx: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply the chain rule using precomputed Jacobians."""
        if ref_idx is None:
            raise ValueError("`ref_idx` is required.")

        ref_idx = ref_idx.flatten().to(
            self.jacobian.device,
            dtype=torch.long,
        )
        if torch.any((ref_idx < 0) | (ref_idx >= len(self.jacobian))):
            raise IndexError("Invalid Jacobian index.")

        jacobian = self.jacobian[ref_idx].to(
            gradient.device,
            gradient.dtype,
        )
        return torch.einsum("bl,b...l->b...", gradient, jacobian)