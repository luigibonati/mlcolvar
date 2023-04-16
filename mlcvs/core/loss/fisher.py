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
            invert_sign: bool = True):
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
        invert_sign: bool, optional
            Whether to return the negative Fisher's discriminant ratio in order to be
            minimized with gradient descent methods. Default is ``True``.
        """
        super().__init__()
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
        return fisher_discriminant_loss(x, labels, self.invert_sign)


def fisher_discriminant_loss(
        x: torch.Tensor,
        labels: torch.Tensor,
        n_states: int,
        lda_mode: str = 'standard',
        reduce_mode: str = 'sum',
        invert_sign: bool = True,
) -> torch.Tensor:
    """Fisher's discriminant ratio.

    Computes the sum (or another reducing functions) of the eigenvalues of the
    ratio between the Fisher's scatter matrices. This is the same loss function
    used in :class:`~mlcvs.cvs.supervised.deeplda.DeepLDA`.
    
    Parameters
    ----------
    x : torch.Tensor
        Shape ``(n_batches, n_features)``. Input features.
    labels : torch.Tensor
        Shape ``(n_batches,)``. Classes labels.
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
    invert_sign: bool, optional
        Whether to return the negative Fisher's discriminant ratio in order to be
        minimized with gradient descent methods. Default is ``True``.

    Returns
    -------
    loss: torch.Tensor
        Loss value.
    """
    lda = LDA(in_features=x.shape[-1], n_states=n_states, mode=lda_mode)
    eigvals, _ = lda.compute(x, labels)
    loss = reduce_eigenvalues_loss(eigvals, mode=reduce_mode, invert_sign=invert_sign)
    return loss
