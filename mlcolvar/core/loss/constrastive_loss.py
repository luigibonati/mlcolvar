#!/usr/bin/env python

# =============================================================================
# MODULE DOCSTRING
# =============================================================================

"""
Contrastive loss module.

This module implements a loss combining a contrastive 
loss term with a regularization on the input
"""

__all__ = ["ContrastiveLoss", "contrastive_loss"]

# =============================================================================
# GLOBAL IMPORTS
# =============================================================================

import torch
import math

# =============================================================================
# LOSS FUNCTIONS
# =============================================================================


class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive Loss module.

    This loss combines a contrastive spectral objective with an L2 regularization
    term on the learned representations.

    Parameters
    ----------
    mode : str, optional
        Contrastive loss type: {"l2", "kl_DV", "kl_NWJ"}.
    reg : float, optional
        Regularization coefficient (default: 1e-5).
    """

    def __init__(self, mode: str = "l2", reg: float = 1e-5):
        super().__init__()
        self.mode = mode
        self.reg = reg

    def forward(
        self, 
        inputs: torch.Tensor, 
        lagged: torch.Tensor,
        remove_average: bool = True,
    ) -> torch.Tensor:
        """
        Compute the regularized spectral loss.

        Parameters
        ----------
        inputs : torch.Tensor
            Representations at time t.
        lagged : torch.Tensor
            Representations at time t+τ.
        remove_average : bool, optional
            Whether to subtract the mean from the input representations
            before computing time-correlation matrices.

        Returns
        -------
        torch.Tensor
            Scalar loss value.
        """
        return contrastive_loss(
            inputs,
            lagged,
            reg=self.reg,
            mode=self.mode,
            remove_average=remove_average,
        )

    def noreg(
        self, 
        inputs: torch.Tensor, 
        lagged: torch.Tensor,
        remove_average=True
    ) -> torch.Tensor:
        """
        Compute the contrastive term only (without regularization).
        """
        return contrastive_loss(
            inputs,
            lagged,
            reg=0.0,
            mode=self.mode,
            remove_average=remove_average,
        )
    

def contrastive_loss(
    x: torch.Tensor,
    y: torch.Tensor,
    mode: str = "l2",
    reg: float = 1e-5,
    remove_average: bool = True,
) -> torch.Tensor:
    """
    Compute the contrastive loss.

    This function unifies the contrastive spectral objectives used in SelfTICA
    and related methods. Depending on the chosen `mode`, it computes:

    - L2 decorrelation loss (closely related to the VAMP-2 score)
    - KL-based loss via Donsker-Varadhan bound
    - KL-based loss via Nguyen-Wainwright-Jordan bound

    Optionally, an L2 penalty on the feature norms can be added via `reg`.

    Parameters
    ----------
    x, y : torch.Tensor
        Input tensors of shape (n_samples, n_features), representing
        configurations at time t and t+τ.
    mode : str, optional
        Contrastive loss type: {"l2", "kl_DV", "kl_NWJ"}.
    reg : float, optional
        Regularization strength.
    remove_average : bool, optional
        Whether to subtract the (weighted) mean from the input representations
        before computing time-correlation matrices.

    Returns
    -------
    torch.Tensor
        Scalar loss value.
    """
    if x.shape != y.shape:
        raise ValueError("Inputs must have the same shape.")
    if x.ndim != 2:
        raise ValueError("Inputs must be 2D tensors.")

    npts, dim = x.shape
    if npts < 2:
        raise ValueError("Spectral loss requires at least 2 samples.")
    
    # remove mean
    if remove_average:
        x = x - torch.mean(x, dim=0)
        y = y - torch.mean(y, dim=0)

    # similarity matrix
    sim_mat = torch.matmul(x, y.T)

    # remove diagonal terms (positive pairs)
    sim_mat_nodiag = torch.triu(sim_mat, diagonal=1) + torch.tril(
        sim_mat, diagonal=-1
    )

    # positive term
    pos_term = torch.mean(x * y) * dim

    # -------------------------
    # Contrastive loss term
    # -------------------------
    if mode == "l2":
        diag = 2.0 * pos_term
        neg_term = (sim_mat_nodiag**2).mean() * npts / (npts - 1)
        loss = neg_term - diag

    elif mode == "kl_DV":
        log_term = torch.logsumexp(sim_mat_nodiag, dim=(0, 1))
        log_term = log_term - math.log(npts * (npts - 1))
        loss = log_term - pos_term

    elif mode == "kl_NWJ":
        exp_term = (sim_mat_nodiag - 1.0).exp().mean() * npts / (npts - 1)
        loss = exp_term - pos_term

    else:
        raise ValueError(f"Unknown mode '{mode}'. Supported: l2, kl_DV, kl_NWJ")

    # regularization
    if reg > 0.0:
        x_norm2 = torch.linalg.matrix_norm(x, ord="fro") ** 2
        y_norm2 = torch.linalg.matrix_norm(y, ord="fro") ** 2
        loss = loss + reg * (x_norm2 + y_norm2) / 2.0

    return loss