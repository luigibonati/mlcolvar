#!/usr/bin/env python

# =============================================================================
# MODULE DOCSTRING
# =============================================================================

"""
Fisher discriminant loss for (Deep) Linear Discriminant Analysis.
"""

__all__ = ['FisherDiscriminantLoss', 'fisher_discriminant_loss']


# =============================================================================
# GLOBAL IMPORTS
# =============================================================================

from typing import Optional

import torch

from mlcvs.core.stats import LDA
from mlcvs.core.loss import reduce_eigenvalues_loss


# =============================================================================
# LOSS FUNCTIONS
# =============================================================================

class FisherDiscriminantLoss(torch.nn.Module):
    """Fisher's discriminant ratio.

    Computes the sum (or another reducing functions) of the eigenvalues of the
    ratio between the Fisher's scatter matrices. This is the same loss function
    used in :class:`~mlcvs.cvs.supervised.deeplda.DeepLDA`.
    """

    def __init__(
            self,
            n_states: int,
            lda_mode: str = 'standard',
            reduce_mode: str = 'sum',
            lorentzian_reg: Optional[float] = None,
            invert_sign: bool = True
    ):
        """Constructor.

        Parameters
        ----------
        n_states : int
            The number of states. Labels are in the range ``[0, n_states-1]``.
        lda_mode : str
            Either ``'standard'`` or ``'harmonic'``. This determines how the scatter
            matrices are computed (see also :class:`~mlcvs.core.stats.lda.LDA`). The
            default is ``'standard'``.
        reduce_mode : str
            This determines how the eigenvalues are reduced, e.g., ``sum``, ``sum2``
            (see also :class:`~mlcvs.core.loss.eigvals.ReduceEigenvaluesLoss`). The
            default is ``'sum'``.
        lorentzian_reg: float, optional
            The magnitude of the regularization for Lorentzian regularization.
            If not provided, this is automatically set.
        invert_sign: bool, optional
            Whether to return the negative Fisher's discriminant ratio in order to be
            minimized with gradient descent methods. Default is ``True``.
        """
        super().__init__()
        self.n_states = n_states
        self.lda_mode = lda_mode
        self.reduce_mode = reduce_mode
        self.lorentzian_reg = lorentzian_reg
        self.invert_sign = invert_sign

    def forward(
            self,
            x: torch.Tensor,
            labels: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the value of the loss function.

        Parameters
        ----------
        x : torch.Tensor
            Shape ``(n_batches, n_features)``. Input features.
        labels : torch.Tensor
            Shape ``(n_batches,)``. Classes labels.

        Returns
        -------
        loss : torch.Tensor
            Loss value.
        """
        return fisher_discriminant_loss(
            x, labels,
            n_states=self.n_states,
            lda_mode=self.lda_mode,
            reduce_mode=self.reduce_mode,
            lorentzian_reg=self.lorentzian_reg,
            invert_sign=self.invert_sign
        )


def fisher_discriminant_loss(
        x: torch.Tensor,
        labels: torch.Tensor,
        n_states: int,
        lda_mode: str = 'standard',
        reduce_mode: str = 'sum',
        sw_reg: Optional[float] = 0.05,
        lorentzian_reg: Optional[float] = None,
        invert_sign: bool = True,
) -> torch.Tensor:
    """Fisher's discriminant ratio.

    Computes the sum (or another reducing functions) of the eigenvalues of the
    ratio between the Fisher's scatter matrices with a Lorentzian regularization.
    This is the same loss function used in :class:`~mlcvs.cvs.supervised.deeplda.DeepLDA`.
    
    Parameters
    ----------
    x : torch.Tensor
        Shape ``(n_batches, n_features)``. Input features.
    labels : torch.Tensor
        Shape ``(n_batches,)``. Classes labels.
    n_states : int
        The number of states. Labels are in the range ``[0, n_states-1]``.
    lda_mode : str, optional
        Either ``'standard'`` or ``'harmonic'``. This determines how the scatter
        matrices are computed (see also :class:`~mlcvs.core.stats.lda.LDA`). The
        default is ``'standard'``.
    reduce_mode : str, optional
        This determines how the eigenvalues are reduced, e.g., ``sum``, ``sum2``
        (see also :class:`~mlcvs.core.loss.eigvals.ReduceEigenvaluesLoss`). The
        default is ``'sum'``.
    sw_reg: float, optional
        The magnitude of the regularization for the within-scatter matrix, by default
        equal to 0.05.
    lorentzian_reg: float, optional
        The magnitude of the regularization for Lorentzian regularization. If not
        provided, this is automatically set according to sw_reg.
    invert_sign: bool, optional
        Whether to return the negative Fisher's discriminant ratio in order to be
        minimized with gradient descent methods. Default is ``True``.

    Returns
    -------
    loss: torch.Tensor
        Loss value.
    """
    # define lda object
    lda = LDA(in_features=x.shape[-1], n_states=n_states, mode=lda_mode)

    # regularize s_w
    lda.sw_reg = sw_reg

    # compute LDA eigvals
    eigvals, _ = lda.compute(x, labels)
    loss = reduce_eigenvalues_loss(eigvals, mode=reduce_mode, invert_sign=invert_sign)

    # Add lorentzian regularization. The heuristic is the same used by DeepLDA.
    # TODO: ENCAPSULATE THIS IN A UTILITY FUNCTION USED BY BOTH THIS AND DEEPLDA?
    if lorentzian_reg is None:
        if sw_reg == 0 or sw_reg is None:
            raise ValueError(f'Unable to calculate `lorentzian_reg` from `sw_reg` ({sw_reg}), please specify the value.')
        lorentzian_reg = 2.0 / sw_reg
    reg_loss = x.pow(2).sum().div(x.size(0))
    reg_loss = - lorentzian_reg / (1 + (reg_loss - 1).pow(2))

    return loss + reg_loss
