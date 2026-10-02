from typing import Optional

import torch
from torch import nn

__all__ = ["CachedRepresentationDerivatives"]


class CachedRepresentationDerivatives(nn.Module):
    """Apply cached representation Jacobians through the chain rule.

    Given gradients with respect to latent representation features,
    transform them to gradients with respect to the original coordinates
    using cached representation Jacobians.

    Parameters
    ----------
    jacobian : torch.Tensor
        Cached representation Jacobians. The last dimension corresponds
        to latent representation features.
    """

    def __init__(self, jacobian: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("jacobian", jacobian, persistent=False)

    def forward(
        self,
        gradient_latent: torch.Tensor,
        ref_idx: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Transform latent gradients using cached Jacobians.

        Parameters
        ----------
        gradient_latent : torch.Tensor
            Gradients with respect to latent representation features.
        ref_idx : torch.Tensor
            Indices mapping the current batch to cached Jacobians.

        Returns
        -------
        torch.Tensor
            Gradients with respect to the original coordinates.
        """
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