from contextlib import contextmanager
from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
from torch_geometric.loader import DataLoader as GraphDataLoader

from mlcolvar.core.loss.utils.smart_derivatives import SmartDerivatives
from mlcolvar.data import DictDataset

from ._utils import module_reference_tensor
from .base import Representation

__all__ = [
    "RepresentationCache",
    "IdentityDescriptorDerivatives",
    "CachedRepresentationDerivatives",
    "precompute_representation_cache",
]


@dataclass
class RepresentationCache:
    """Cached outputs of a frozen representation."""

    features: torch.Tensor
    jacobian: Optional[torch.Tensor] = None
    reference_indices: Optional[torch.Tensor] = None


class CachedRepresentationDerivatives(SmartDerivatives):
    """Apply cached representation Jacobians through the chain rule."""

    def __init__(self, jacobian: torch.Tensor):
        nn.Module.__init__(self)
        self.register_buffer("jacobian", jacobian, persistent=False)

    def forward(self, gradient_latent, ref_idx=None):
        if ref_idx is None:
            raise ValueError("`ref_idx` is required.")
        ref_idx = ref_idx.reshape(-1).to(
            self.jacobian.device,
            dtype=torch.long,
        )
        if torch.any(ref_idx < 0) or torch.any(ref_idx >= len(self.jacobian)):
            raise IndexError("Invalid cached derivative index.")
        jacobian = self.jacobian[ref_idx].to(
            device=gradient_latent.device,
            dtype=gradient_latent.dtype,
        )
        return torch.einsum(
            "bl,b...l->b...",
            gradient_latent,
            jacobian,
        )
        

class IdentityDescriptorDerivatives(nn.Module):
    """Treat input descriptors as Cartesian coordinates."""

    def __init__(
        self,
        n_atoms: int,
        n_dim: int,
    ) -> None:
        super().__init__()
        self.n_atoms = int(n_atoms)
        self.n_dim = int(n_dim)

    def forward(
        self,
        gradient_descriptor: torch.Tensor,
        ref_idx=None,
    ) -> torch.Tensor:
        expected = self.n_atoms * self.n_dim

        if gradient_descriptor.shape[-1] != expected:
            raise ValueError(
                f"Expected {expected} Cartesian descriptors, "
                f"found {gradient_descriptor.shape[-1]}."
            )

        return gradient_descriptor.reshape(
            gradient_descriptor.shape[0],
            self.n_atoms,
            self.n_dim,
        )


def _indices(n, indices=None):
    """Normalize selected sample indices."""
    if indices is None:
        return torch.arange(n, dtype=torch.long)
    indices = torch.as_tensor(indices).cpu()
    if indices.dtype == torch.bool:
        indices = indices.reshape(-1)
        if len(indices) != n:
            raise ValueError(
                "Boolean `jacobian_indices` must have the same length "
                "as the dataset."
            )
        indices = torch.nonzero(indices, as_tuple=False).reshape(-1)
    else:
        indices = indices.reshape(-1).long()
    indices = torch.unique(indices, sorted=True)
    if torch.any(indices < 0) or torch.any(indices >= n):
        raise IndexError(
            "`jacobian_indices` contains out-of-range indices."
        )
    return indices


def _references(n, selected, device):
    """Map dataset indices to cached Jacobian indices."""
    refs = torch.full((n,), -1, dtype=torch.long, device=device)
    selected = selected.to(device)
    refs[selected] = torch.arange(len(selected), device=device)
    return refs


@contextmanager
def _temporary_eval(representation, device):
    """Temporarily evaluate a representation on another device."""
    original_device = module_reference_tensor(representation).device
    training = representation.training
    representation.to(device).eval()
    try:
        yield
    finally:
        representation.to(original_device).train(training)


def _cache_vector(
    representation,
    dataset,
    *,
    descriptor_derivatives,
    jacobian_indices,
    source_ref_idx,
    batch_size,
    device,
    output_device,
    compute_jacobian,
):
    """Cache features and optional Jacobians for vector inputs."""
    if "data" not in dataset.keys:
        raise KeyError("Vector caching requires `data`.")
    x = dataset["data"]
    cell = dataset["cell"] if "cell" in dataset.keys else None
    n = len(x)

    with _temporary_eval(representation, device):
        features = []
        with torch.no_grad():
            for start in range(0, n, batch_size):
                stop = min(start + batch_size, n)
                xb = x[start:stop].to(device)
                cb = None if cell is None else cell[start:stop].to(device)
                h = representation(xb, cell=cb)
                features.append(
                    h.reshape(len(xb), -1).detach().to(output_device)
                )
        features = torch.cat(features)

        if not compute_jacobian:
            return RepresentationCache(features)
        if descriptor_derivatives is None:
            raise ValueError("`descriptor_derivatives` is required.")

        descriptor_derivatives.to(device)
        selected = _indices(n, jacobian_indices)
        if selected.numel() == 0:
            raise ValueError("No samples selected for Jacobian caching.")

        source_ref_idx = (
            torch.arange(n)
            if source_ref_idx is None
            else torch.as_tensor(source_ref_idx).reshape(-1).long().cpu()
        )

        jacobians = []
        with torch.enable_grad():
            for start in range(0, len(selected), batch_size):
                idx = selected[start:start + batch_size]
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
                refs = source_ref_idx[idx].to(device)
                h = representation(xb, cell=cb).reshape(len(xb), -1)

                grads = []
                for i in range(h.shape[-1]):
                    dh = torch.autograd.grad(
                        h[:, i].sum(),
                        xb,
                        retain_graph=i + 1 < h.shape[-1],
                    )[0]
                    grads.append(descriptor_derivatives(dh, refs))

                jacobians.append(
                    torch.stack(grads, dim=-1)
                    .detach()
                    .to(output_device)
                )

        return RepresentationCache(
            features=features,
            jacobian=torch.cat(jacobians),
            reference_indices=_references(
                n,
                selected,
                output_device,
            ),
        )


def _cache_graph(
    representation,
    dataset,
    *,
    jacobian_indices,
    batch_size,
    device,
    output_device,
    compute_jacobian,
):
    """Cache features and optional Cartesian Jacobians for graph inputs."""
    if dataset.metadata.get("data_type") != "graphs":
        raise TypeError("Expected a graph dataset.")

    n = len(dataset)
    selected = (
        _indices(n, jacobian_indices)
        if compute_jacobian
        else None
    )
    if compute_jacobian and selected.numel() == 0:
        raise ValueError("No samples selected for Jacobian caching.")

    with _temporary_eval(representation, device):
        loader = GraphDataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
        )
        features = []
        jacobians = []
        offset = 0
        n_atoms_ref = None

        for batch in loader:
            graph = batch["data_list"].to(device)
            n_graphs = int(graph.num_graphs)
            local = (
                selected[
                    (selected >= offset)
                    & (selected < offset + n_graphs)
                ] - offset
                if compute_jacobian
                else torch.empty(0, dtype=torch.long)
            )
            need_grad = compute_jacobian and len(local) > 0

            if need_grad:
                counts = torch.bincount(
                    graph.batch,
                    minlength=n_graphs,
                )
                selected_counts = counts[local.to(counts.device)]
                if not torch.all(
                    selected_counts == selected_counts[0]
                ):
                    raise ValueError(
                        "Selected graphs must have equal atom counts."
                    )

                n_atoms = int(selected_counts[0])
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

            with (
                torch.enable_grad()
                if need_grad
                else torch.no_grad()
            ):
                h = representation(graph)
                if h.ndim != 2 or h.shape[0] != n_graphs:
                    raise ValueError(
                        "Cached features must contain one row per graph."
                    )

                if need_grad:
                    local = local.to(device)
                    hs = h[local]
                    grads = []
                    for i in range(h.shape[-1]):
                        dp = torch.autograd.grad(
                            hs[:, i].sum(),
                            graph.positions,
                            retain_graph=i + 1 < h.shape[-1],
                        )[0]
                        grads.append(
                            torch.stack(
                                [
                                    dp[graph.batch == j]
                                    for j in local
                                ]
                            )
                        )

                    jacobians.append(
                        torch.stack(grads, dim=-1)
                        .detach()
                        .to(output_device)
                    )

            features.append(h.detach().to(output_device))
            offset += n_graphs

        features = torch.cat(features)
        if not compute_jacobian:
            return RepresentationCache(features)

        return RepresentationCache(
            features=features,
            jacobian=torch.cat(jacobians),
            reference_indices=_references(
                n,
                selected,
                output_device,
            ),
        )


def precompute_representation_cache(
    representation: Representation,
    dataset: DictDataset,
    *,
    descriptor_derivatives: Optional[SmartDerivatives] = None,
    jacobian_indices: Optional[torch.Tensor] = None,
    source_ref_idx: Optional[torch.Tensor] = None,
    batch_size: Optional[int] = None,
    device=None,
    output_device="cpu",
    compute_jacobian: bool = False,
):
    """Cache a frozen representation and optional Jacobians.

    Parameters
    ----------
    representation : Representation
        Frozen representation to evaluate.
    dataset : DictDataset
        Dataset containing raw vector or graph inputs.
    descriptor_derivatives : SmartDerivatives, optional
        Descriptor derivatives used for vector Jacobian caching.
    jacobian_indices : torch.Tensor, optional
        Samples for which Jacobians are cached.
    source_ref_idx : torch.Tensor, optional
        Reference indices used by ``descriptor_derivatives``.
    batch_size : int, optional
        Evaluation batch size.
    device : str or torch.device, optional
        Device used to evaluate the representation.
    output_device : str or torch.device, default="cpu"
        Device used to store cached tensors.
    compute_jacobian : bool, default=False
        Whether to cache representation Jacobians.

    Returns
    -------
    RepresentationCache
        Cached representation outputs and optional Jacobians.
    """
    if not isinstance(representation, Representation):
        raise TypeError(
            "`representation` must derive from `Representation`."
        )
    if not representation.freeze:
        raise RuntimeError(
            "Caching requires a frozen representation."
        )

    device = torch.device(
        device
        or module_reference_tensor(representation).device
    )
    output_device = torch.device(output_device)

    if representation.input_kind == "vector":
        return _cache_vector(
            representation,
            dataset,
            descriptor_derivatives=descriptor_derivatives,
            jacobian_indices=jacobian_indices,
            source_ref_idx=source_ref_idx,
            batch_size=batch_size or 1024,
            device=device,
            output_device=output_device,
            compute_jacobian=compute_jacobian,
        )

    if representation.input_kind == "graph":
        if representation.output_kind != "system":
            raise ValueError(
                "Graph caching requires system-level representation outputs. "
                "Use pooling or `concat_atoms()` first."
            )
        if (
            descriptor_derivatives is not None
            or source_ref_idx is not None
        ):
            raise ValueError(
                "Descriptor derivatives and source indices are vector-only."
            )

        return _cache_graph(
            representation,
            dataset,
            jacobian_indices=jacobian_indices,
            batch_size=batch_size or 256,
            device=device,
            output_device=output_device,
            compute_jacobian=compute_jacobian,
        )

    raise ValueError(
        "Unsupported representation input kind: "
        f"{representation.input_kind!r}."
    )