#!/usr/bin/env python

# =============================================================================
# MODULE DOCSTRING
# =============================================================================

"""
(Weighted) Mean Squared Error (MSE) loss function.
"""

__all__ = ["MSELoss", "mse_loss"]


# =============================================================================
# GLOBAL IMPORTS
# =============================================================================

from typing import Optional

import torch


# =============================================================================
# LOSS FUNCTIONS
# =============================================================================


class MSELoss(torch.nn.Module):
    """(Weighted) Mean Square Error"""

    def forward(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute the value of the loss function."""
        return mse_loss(input, target, weights)


def mse_loss(
    input: torch.Tensor, target: torch.Tensor, weights: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """(Weighted) Mean Square Error

    Parameters
    ----------
    input : torch.Tensor
        prediction
    target : torch.Tensor
        reference
    weights : torch.Tensor, optional
        sample weights, by default None

    Returns
    -------
    loss: torch.Tensor
        loss function
    """
    # reshape in the correct format (batch, size)
    if input.ndim == 1:
        input = input.unsqueeze(1)
    if target.ndim == 1:
        target = target.unsqueeze(1)
    # take the different
    diff = input - target
    # weight them
    if weights is not None:
        if weights.ndim == 1:
            weights = weights.unsqueeze(1)
        loss = (diff * weights).square().mean()
    else:
        loss = diff.square().mean()
    return loss
