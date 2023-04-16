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

from mlcvs.core.stats.tica import TICA
from mlcvs.core.loss.eigvals import reduce_eigenvalues_loss


# =============================================================================
# LOSS FUNCTIONS
# =============================================================================

class AutocorrelationLoss(torch.nn.Module):
    """(Weighted) autocorrelation loss.

    Computes the sum (or another reducing functions) of the eigenvalues of the
    autocorrelation matrix. This is the same loss function used in
    :class:`~mlcvs.cvs.timelagged.deeptica.DeepTICA`.

    """

    def __init__(self, reduce_mode: str = 'sum2', invert_sign: bool = True):
        """Constructor.

        Parameters
        ----------
        reduce_mode : str
            This determines how the eigenvalues are reduced, e.g., ``sum``, ``sum2``
            (see also :class:`~mlcvs.core.loss.eigvals.ReduceEigenvaluesLoss`). The
            default is ``'sum2'``.
        invert_sign: bool, optional
            Whether to return the negative autocorrelation in order to be minimized
            with gradient descent methods. Default is ``True``.
        """
        super().__init__()
        self.reduce_mode = reduce_mode
        self.invert_sign = invert_sign

    def forward(
            self,
            x: torch.Tensor,
            x_lag: torch.Tensor,
            weights: Optional[torch.Tensor] = None,
            weights_lag: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Estimate the autocorrelation.

        Parameters
        ----------
        x : torch.Tensor
            Shape ``(n_batches, n_features)``. The features of the sample at
            time ``t``.
        x_lag : torch.Tensor
            Shape ``(n_batches, n_features)``. The features of the sample at
            time ``t + lag``.
        weights : torch.Tensor, optional
            Shape ``(n_batches,)`` or ``(n_batches, 1)``. The weights associated
            to ``x`` at time ``t``. Default is ``None``.
        weights_lag : torch.Tensor, optional
            Shape ``(n_batches,)`` or ``(n_batches, 1)``. The weights associated
            to ``x`` at time ``t + lag``. Default is ``None``.

        Returns
        -------
        loss : torch.Tensor
            Loss value.
        """
        return autocorrelation_loss(
            x, x_lag,
            weights=weights,
            weights_lag=weights_lag,
            reduce_mode=self.reduce_mode,
            invert_sign=self.invert_sign,
        )


def autocorrelation_loss(
        x: torch.Tensor,
        x_lag: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
        weights_lag: Optional[torch.Tensor] = None,
        reduce_mode: str = 'sum2',
        invert_sign: bool = True,
) -> torch.Tensor:
    """(Weighted) autocorrelation loss.

    Computes the sum (or another reducing functions) of the eigenvalues of the
    autocorrelation matrix. This is the same loss function used in
    :class:`~mlcvs.cvs.timelagged.deeptica.DeepTICA`.
    
    Parameters
    ----------
    x : torch.Tensor
        Shape ``(n_batches, n_features)``. The features of the sample at
        time ``t``.
    x_lag : torch.Tensor
        Shape ``(n_batches, n_features)``. The features of the sample at
        time ``t + lag``.
    weights : torch.Tensor, optional
        Shape ``(n_batches,)`` or ``(n_batches, 1)``. The weights associated
        to ``x`` at time ``t``. Default is ``None``.
    weights_lag : torch.Tensor, optional
        Shape ``(n_batches,)`` or ``(n_batches, 1)``. The weights associated
        to ``x`` at time ``t + lag``. Default is ``None``.
    reduce_mode : str
        This determines how the eigenvalues are reduced, e.g., ``sum``, ``sum2``
        (see also :class:`~mlcvs.core.loss.eigvals.ReduceEigenvaluesLoss`). The
        default is ``'sum2'``.
    invert_sign: bool, optional
        Whether to return the negative autocorrelation in order to be minimized
        with gradient descent methods. Default is ``True``.

    Returns
    -------
    loss: torch.Tensor
        Loss value.
    """
    tica = TICA(in_features=x.shape[-1])
    eigvals, _ = tica.compute(data=[x, x_lag], weights=[weights, weights_lag])
    loss = reduce_eigenvalues_loss(eigvals, mode=reduce_mode, invert_sign=invert_sign)
    return loss
