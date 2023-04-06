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


# =============================================================================
# LOSS FUNCTIONS
# =============================================================================

class FisherDiscriminantLoss(torch.nn.Module):
    """ Fisher's discriminant ratio.

    .. math::
        L = - \frac{S_b(X)}{S_w(X)}
    """

    def __init__(self, invert_sign : bool = True):
        super().__init__()
        self.invert_sign = invert_sign

    def forward(
            self,
            X: torch.Tensor,
            labels: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the value of the loss function.

        Parameters
        ----------
        X : torch.Tensor
            Shape ``(n_batches, n_features)``. Input features.
        labels : torch.Tensor
            Shape ``(n_batches,)``. Classes labels.

        Returns
        -------
        loss : torch.Tensor
            Loss value.
        """
        return fisher_discriminant_loss(X, labels. self.invert_sign)


def fisher_discriminant_loss(
        X: torch.Tensor,
        labels: torch.Tensor,
        invert_sign: bool = True
) -> torch.Tensor:
    """ Fisher's discriminant ratio.

    .. math::
        L = - \frac{S_b(X)}{S_w(X)}
    
    Parameters
    ----------
    X : torch.Tensor
        Shape ``(n_batches, n_features)``. Input features.
    labels : torch.Tensor
        Shape ``(n_batches,)``. Classes labels.
    invert_sign: bool, optional
        whether to return the negative Fisher's discriminant ratio in order to be
        minimized with gradient descent methods. Default is ``True``.

    Returns
    -------
    loss: torch.Tensor
        Loss value.
    """

    if X.ndim == 1:
        X = X.unsqueeze(1) # for lda compute_scatter_matrices method
    if X.ndim == 2 and X.shape[1] > 1:
        raise ValueError (f'fisher_discriminant_loss should be used on (batches of) scalar outputs, found tensor of shape {X.shape} instead.')

    # get params
    d = X.shape[-1] if X.ndim == 2 else 1
    n_classes = len(labels.unique())

    # define LDA object to compute S_b / S_w ratio
    lda = LDA(in_features=d, n_states=n_classes)

    s_b, s_w = lda.compute_scatter_matrices(X,labels)

    loss = s_b.squeeze() / s_w.squeeze()

    if invert_sign:
        loss *= -1

    return loss
