#!/usr/bin/env python

# =============================================================================
# MODULE DOCSTRING
# =============================================================================

"""
Regularized spectral loss module.

This module implements a regularized spectral loss combining a 
contrastive loss term with a regularization on the input
"""

__all__ = ["RegSpectralLoss", "reg_spectral_loss"]

# =============================================================================
# GLOBAL IMPORTS
# =============================================================================

import torch


# =============================================================================
# LOSS FUNCTIONS
# =============================================================================


class RegSpectralLoss(torch.nn.Module):
    """ 
    Regularized Spectral Loss Module.

    Combined an L2 contractive loss term with a regularization on the inputs. 

    Parameters
    ----------
    reg : float, optional
        Regularization coefficient..
    """
    def __init__(self, reg: float = 1e-5):
        super().__init__()
        self.reg = reg
    
    def forward(self, inputs, lagged):
        """
        Compute the regularized spectral loss. 

        Parameters
        ----------
        inputs: torch.Tensor
            Current inputs tensor of shape (n_samples, features).
        lagged: torch.Tensor
            Lagged inputs tensor of shape (n_samples, features).
        """
        return reg_spectral_loss(inputs, lagged, self.reg)
    
    def noreg(self, inputs, lagged):
        return l2_contrastive_loss(inputs, lagged)
    

def l2_contrastive_loss(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """ 
    Compute the L2 contrastive loss to encourage both temporal consistency 
    and decorrelation between input representations.

    Parameters
    ----------
    x, y : torch.Tensor
        Input tensors of shape (n_samples, features)

    Returns
    -------
    loss : torch.Tensor (scalar)
          A scalar tensor representing the contrastive loss.
    """

    assert x.shape == y.shape, "Input must have the same shape"
    assert x.ndim == 2, "Inputs must be 2D tensors"

    npts, dim = x.shape
    diag = 2 * torch.mean(x * y) * dim
    square_term = torch.matmul(x, y.T) ** 2
    off_diag = (
        torch.mean(torch.triu(square_term, diagonal=1) + torch.tril(square_term, diagonal=-1))
        * npts
        / (npts - 1)
    )

    return  off_diag - diag


def reg_spectral_loss(
    inputs: torch.Tensor,
    lagged: torch.Tensor,
    reg: float = 1e-5,
) -> torch.Tensor:
    """ 
    Compute the regularized spectral loss combining L2 contrastive loss
    and a regularization term on the inputs.

    Parameters
    ----------
    inputs : torch.Tensor
        Current inputs tensor of shape (n_samples, features).
    lagged : torch.Tensor
        Lagged inputs tensor of shape (n_samples, features).
    reg: float, optional

    Returns
    -------
    torch.Tensor
        Scalar loss value.
    """
    
    loss = l2_contrastive_loss(inputs, lagged)
    inputs_norm2 = torch.linalg.matrix_norm(inputs) ** 2
    lagged_norm2 = torch.linalg.matrix_norm(lagged) ** 2
    reg_term = reg * (inputs_norm2 + lagged_norm2) / 2

    return loss + reg_term

    
