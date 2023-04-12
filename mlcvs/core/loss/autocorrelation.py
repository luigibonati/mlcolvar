#!/usr/bin/env python

# =============================================================================
# MODULE DOCSTRING
# =============================================================================

"""
Autocorrelation loss.
"""

__all__ = ['AutocorrelationLoss', 'autocorrelation_loss']


# =============================================================================
# GLOBAL IMPORTS
# =============================================================================

from typing import Optional

import torch


# =============================================================================
# LOSS FUNCTIONS
# =============================================================================

class AutocorrelationLoss(torch.nn.Module):
    r"""(Weighted) autocorrelation loss.

    .. math::

        L = - \frac{\langle (x(t)-\bar{x}(t))(x(t+\tau)-\bar{x}(t)) \rangle}{\sigma(x_t)^2}

    """

    def __init__(self, invert_sign: bool = True):
        """Constructor.

        Parameters
        ----------
        invert_sign: bool, optional
            Whether to return the negative autocorrelation in order to be minimized
            with gradient descent methods. Default is ``True``.
        """
        super().__init__()
        self.invert_sign = invert_sign

    def forward(
            self,
            x_t: torch.Tensor,
            x_lag: torch.Tensor,
            weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Estimate the autocorrelation.

        Parameters
        ----------
        x_t : torch.Tensor
            Shape ``(n_batches, n_features)``. The features of the sample at
            time ``t``.
        x_lag : torch.Tensor
            Shape ``(n_batches, n_features)``. The features of the sample at
            time ``t + lag``.
        weights : torch.Tensor, optional
            Shape ``(n_batches,)`` or ``(n_batches, 1)``. The weight associated
            to each batch sample. Default is ``None``.

        Returns
        -------
        loss : torch.Tensor
            Loss value.
        """
        return autocorrelation_loss(x_t, x_lag, weights=weights, invert_sign=self.invert_sign)


def autocorrelation_loss(
        x_t: torch.Tensor,
        x_lag: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
        invert_sign: bool = True,
) -> torch.Tensor:
    r"""(Weighted) autocorrelation loss.

    .. math::

        L = - \frac{\langle (x(t)-\bar{x}(t))(x(t+\tau)-\bar{x}(t)) \rangle}{\sigma(x_t)^2}
    
    Parameters
    ----------
    x_t : torch.Tensor
        Shape ``(n_batches, n_features)``. The features of the sample at
        time ``t``.
    x_lag : torch.Tensor
        Shape ``(n_batches, n_features)``. The features of the sample at
        time ``t + lag``.
    weights : torch.Tensor, optional
        Shape ``(n_batches,)`` or ``(n_batches, 1)``. The weight associated
        to each batch sample. Default is ``None``.
    invert_sign: bool, optional
        Whether to return the negative autocorrelation in order to be minimized
        with gradient descent methods. Default is ``True``.

    Returns
    -------
    loss: torch.Tensor
        Loss value.
    """
    # Currently we support only 1 dimensional features.
    if x_t.ndim == 2:
        if x_t.shape[1] > 1:
            raise ValueError('The autocorrelation loss should be used on (batches of) scalar '
                             f'outputs, found tensor of shape {z_t.shape} instead.')
        else:
            x_t = x_t.squeeze()
            x_lag = x_lag.squeeze()

    if weights is None:
        mean = x_t.mean()
        std = x_t.std()
        
        loss = ((x_t-mean)*(x_lag-mean)).mean()/std**2
    else:
        weights = weights.squeeze()
        weighted_mean = lambda x,w : (x*w).sum()/w.sum()

        mean = weighted_mean(x_t, weights)
        std = weighted_mean((x_t - mean)**2, weights).sqrt()
        
        loss = weighted_mean((x_t-mean)*(x_lag-mean), weights)/std**2
    
    if invert_sign:
        return -loss

    return loss
