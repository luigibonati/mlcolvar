#!/usr/bin/env python

# =============================================================================
# MODULE DOCSTRING
# =============================================================================

"""
Evidence Lower BOund (ELBO) loss functions used to train variational Autoencoders.
"""

__all__ = ['elbo_gaussians_loss']


# =============================================================================
# GLOBAL IMPORTS
# =============================================================================

from typing import Optional
import torch
from mlcvs.core.loss.mse import mse_loss


# =============================================================================
# LOSS FUNCTIONS
# =============================================================================

def elbo_gaussians_loss(
        target: torch.Tensor,
        output: torch.Tensor,
        mean: torch.Tensor,
        log_variance: torch.Tensor,
        weights: Optional[torch.Tensor] = None
):
    """ELBO loss function assuming the latent and reconstruction distributions are Gaussian.

    The ELBO uses the MSE as the reconstruction loss (i.e., assumes that the
    decoder outputs the mean of a Gaussian distribution with variance 1), and
    the KL divergence between two normal distributions ``N(mean, var)`` and
    ``N(0, 1)``, where ``mean`` and ``var`` are the output of the encoder.

    Parameters
    ----------
    target : torch.Tensor
        Shape ``(n_batches, in_features)``. Data points (e.g. input of encoder or time-lagged features).
    output : torch.Tensor
        Shape ``(n_batches, in_features)``. Output of the decoder.        
    mean : torch.Tensor
        Shape ``(n_batches, latent_features)``. The means of the Gaussian
        distributions associated to the inputs.
    log_variance : torch.Tensor
        Shape ``(n_batches, latent_features)``. The logarithm of the variances
        of the Gaussian distributions associated to the inputs.
    weights : torch.Tensor, optional
        Shape ``(n_batches,)`` or ``(n_batches,1)``. If given, the average over batches is weighted.
        The default (``None``) is unweighted.

    Returns
    -------
    loss: torch.Tensor
        The value of the loss function.
    """
    # KL divergence between N(mean, variance) and N(0, 1).
    # See https://stats.stackexchange.com/questions/7440/kl-divergence-between-two-univariate-gaussians
    kl = -0.5 * (log_variance - log_variance.exp() - mean**2 + 1).sum(dim=1)

    # Weighted mean over batches.
    if weights is None:
        kl = kl.mean()
    else:
        weights = weights.squeeze()
        if weights.shape != kl.shape:
            raise ValueError(f'weights should be a tensor of shape (n_batches,) or (n_batches,1), not {weights.shape}.')
        kl = (kl * weights).sum()

    # Reconstruction loss.
    reconstruction = mse_loss(output, target, weights=weights)

    return reconstruction + kl
