from typing import Optional

import torch
from torch import nn
from torch_geometric.loader import DataLoader as GraphDataLoader

from mlcolvar.data import DictDataset, DictLoader
from mlcolvar.representation._utils import module_reference_tensor

__all__ = ["compute_jacobian", "JacobianTransform"]


def _indices(
    n: int,
    indices: Optional[torch.Tensor] = None,
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
        indices = indices.nonzero().flatten()
    else:
        indices = indices.flatten().long()

    indices = torch.unique(indices, sorted=True)
    if torch.any(indices < 0) or torch.any(indices >= n):
        raise IndexError("`indices` contains out-of-range indices.")

    return indices


def compute_jacobian(
    model: nn.Module,
    dataset: DictDataset,
    *,
    indices: Optional[torch.Tensor] = None,
    batch_size: Optional[int] = None,
    device=None,
    output_device="cpu",
) -> torch.Tensor:
    """Compute model-output Jacobians with respect to dataset inputs.

    For vector datasets, derivatives are computed with respect to ``data``.
    For graph datasets, derivatives are computed with respect to atomic
    positions. Graphs must contain the same number of atoms so that their
    Jacobians can be collected into a dense tensor.

    Parameters
    ----------
    model : torch.nn.Module
        Model whose output Jacobian is evaluated.
    dataset : DictDataset
        Vector- or graph-based dataset containing the model inputs.
    indices : torch.Tensor, optional
        Dataset indices to evaluate. Boolean masks are also supported.
    batch_size : int, optional
        Number of samples evaluated per batch. Defaults to 256 for graph
        datasets and 1024 for vector datasets.
    device : torch.device or str, optional
        Device used for model evaluation. By default, use the current model
        device.
    output_device : torch.device or str, default="cpu"
        Device on which the computed Jacobians are stored.

    Returns
    -------
    torch.Tensor
        Jacobians with the output-feature dimension stored last.

    Notes
    -----
    Samples are assumed to be independent within a batch. This allows the
    gradient of the summed batch output to recover the per-sample Jacobians.
    """
    if not isinstance(model, nn.Module):
        raise TypeError("`model` must be a torch.nn.Module.")

    selected = _indices(len(dataset), indices)
    if selected.numel() == 0:
        raise ValueError("No samples selected for Jacobian computation.")

    dataset = dataset[selected]
    reference = module_reference_tensor(model)
    original_device = reference.device
    training = model.training

    device = torch.device(device or original_device)
    output_device = torch.device(output_device)
    is_graph = dataset.metadata.get("data_type") == "graphs"
    batch_size = batch_size or (256 if is_graph else 1024)

    loader = (
        GraphDataLoader(dataset, batch_size=batch_size, shuffle=False)
        if is_graph
        else DictLoader(dataset, batch_size=batch_size, shuffle=False)
    )

    jacobians = []
    n_atoms_ref = None
    model.to(device).eval()

    try:
        with torch.enable_grad():
            for batch in loader:
                if is_graph:
                    data = batch["data_list"].to(device)
                    n_samples = int(data.num_graphs)

                    # Dense graph Jacobians require a common atom dimension.
                    counts = torch.bincount(data.batch, minlength=n_samples)
                    if not torch.all(counts == counts[0]):
                        raise ValueError("Graphs must have equal atom counts.")

                    n_atoms = int(counts[0])
                    if n_atoms_ref is None:
                        n_atoms_ref = n_atoms
                    elif n_atoms != n_atoms_ref:
                        raise ValueError("Graphs must have equal atom counts.")

                    data.positions = data.positions.detach().requires_grad_(True)
                    output = model(data)
                    inputs = data.positions
                else:
                    data = (
                        batch["data"]
                        .to(device)
                        .detach()
                        .requires_grad_(True)
                    )
                    cell = batch.get("cell")
                    if cell is not None:
                        cell = cell.to(device)

                    n_samples = len(data)
                    output = (
                        model(data)
                        if cell is None
                        else model(data, cell=cell)
                    )
                    inputs = data

                output = output.reshape(n_samples, -1)

                # Summing over independent samples gives the per-sample
                # derivative without constructing the full batch Jacobian.
                gradients = []
                for i in range(output.shape[-1]):
                    gradient = torch.autograd.grad(
                        output[:, i].sum(),
                        inputs,
                        retain_graph=i + 1 < output.shape[-1],
                    )[0]

                    if is_graph:
                        gradient = torch.stack(
                            [
                                gradient[data.batch == j]
                                for j in range(n_samples)
                            ]
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
    """Propagate feature-space gradients through precomputed Jacobians.

    Given a downstream gradient with respect to representation features and
    the corresponding representation Jacobian, this module applies the chain
    rule to obtain gradients with respect to the original inputs.

    Parameters
    ----------
    jacobian : torch.Tensor
        Precomputed representation Jacobians indexed by sample.
    """

    def __init__(self, jacobian: torch.Tensor):
        super().__init__()
        self.register_buffer("jacobian", jacobian, persistent=False)

    def forward(
        self,
        gradient: torch.Tensor,
        ref_idx: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply the chain rule using the selected precomputed Jacobians.

        Parameters
        ----------
        gradient : torch.Tensor
            Gradient of the downstream model with respect to representation
            features.
        ref_idx : torch.Tensor
            Indices selecting the representation Jacobian associated with each
            input sample.

        Returns
        -------
        torch.Tensor
            Gradient propagated to the original input space.
        """
        if ref_idx is None:
            raise ValueError("`ref_idx` is required.")

        ref_idx = ref_idx.flatten().to(
            self.jacobian.device,
            dtype=torch.long,
        )
        if torch.any(ref_idx < 0) or torch.any(ref_idx >= len(self.jacobian)):
            raise IndexError("Invalid Jacobian index.")

        jacobian = self.jacobian[ref_idx].to(
            gradient.device,
            gradient.dtype,
        )
        return torch.einsum("bl,b...l->b...", gradient, jacobian)