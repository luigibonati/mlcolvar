#!/usr/bin/env python

# =============================================================================
# MODULE DOCSTRING
# =============================================================================

"""
(Weighted) Mean Squared Error (MSE) loss function.
"""

__all__ = ['TDALoss', 'tda_loss']


# =============================================================================
# GLOBAL IMPORTS
# =============================================================================

from typing import Union
from warnings import warn

import torch


# =============================================================================
# LOSS FUNCTIONS
# =============================================================================

class TDALoss(torch.nn.Module):
    """Compute a loss function as the distance from a simple Gaussian target distribution."""

    def __init__(
            self,
            n_states: int,
            target_centers: Union[list, torch.Tensor],
            target_sigmas: Union[list, torch.Tensor],
            alfa: float = 1,
            beta: float = 100,
    ):
        """Constructor.

        Parameters
        ----------
        n_states : int
            Number of states. The integer labels are expected to be in between 0
            and ``n_states-1``.
        target_centers : list or torch.Tensor
            Shape ``(n_states, n_cvs)``. Centers of the Gaussian targets.
        target_sigmas : list or torch.Tensor
            Shape ``(n_states, n_cvs)``. Standard deviations of the Gaussian targets.
        alfa : float, optional
            Centers_loss component prefactor, by default 1.
        beta : float, optional
            Sigmas loss compontent prefactor, by default 100.
        """
        super().__init__()
        self.n_states = n_states
        self.target_centers = target_centers
        self.target_sigmas = target_sigmas
        self.alfa = alfa
        self.beta = beta

    def forward(
            self,
            H: torch.Tensor,
            labels: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the value of the loss function.

        Parameters
        ----------
        H : torch.Tensor
            Shape ``(n_batches, n_features)``. Output of the NN.
        labels : torch.Tensor
            Shape ``(n_batches,)``. Labels of the dataset.

        Returns
        -------
        loss : torch.Tensor
            Loss value.
        """
        return tda_loss(H, labels, self.n_states, self.target_centers, self.target_sigmas, self.alfa, self.beta)


def tda_loss(
        H: torch.Tensor,
        labels: torch.Tensor,
        n_states: int,
        target_centers: Union[list, torch.Tensor],
        target_sigmas: Union[list, torch.Tensor],
        alfa: float = 1,
        beta: float = 100,
) -> torch.Tensor:
    """
    Compute a loss function as the distance from a simple Gaussian target distribution.
    
    Parameters
    ----------
    H : torch.Tensor
        Shape ``(n_batches, n_features)``. Output of the NN.
    labels : torch.Tensor
        Shape ``(n_batches,)``. Labels of the dataset.
    n_states : int
        The integer labels are expected to be in between 0 and ``n_states-1``.
    target_centers : list or torch.Tensor
        Shape ``(n_states, n_cvs)``. Centers of the Gaussian targets.
    target_sigmas : list or torch.Tensor
        Shape ``(n_states, n_cvs)``. Standard deviations of the Gaussian targets.
    alfa : float, optional
        Centers_loss component prefactor, by default 1.
    beta : float, optional
        Sigmas loss compontent prefactor, by default 100.

    Returns
    -------
    torch.Tensor
        Total loss, centers loss, and sigmas loss.
    """
    if not isinstance(target_centers, torch.Tensor):
        target_centers = torch.Tensor(target_centers)
    if not isinstance(target_sigmas, torch.Tensor):
        target_sigmas = torch.Tensor(target_sigmas)
    
    loss_centers = torch.zeros_like(target_centers)
    loss_sigmas = torch.zeros_like(target_sigmas)
    for i in range(n_states):
        # check which elements belong to class i
        if not torch.nonzero(labels == i).any():
            raise ValueError(f'State {i} was not represented in this batch! Either use bigger batch_size or a more equilibrated dataset composition!')
        else:
            H_red = H[torch.nonzero(labels == i, as_tuple=True)]

            # compute mean and standard deviation over the class i
            mu = torch.mean(H_red, 0)
            if len(torch.nonzero(labels == i)) == 1:
                warn(f'There is only sample for state {i} in this batch! Std is set to 0, this may affect the training! Either use bigger batch_size or a more equilibrated dataset composition!')
                sigma = 0
            else:
                sigma = torch.std(H_red, 0)

        # compute loss function contributes for class i
        loss_centers[i] = alfa*(mu - target_centers[i]).pow(2)
        loss_sigmas[i] = beta*(sigma - target_sigmas[i]).pow(2)

    # get total model loss   
    loss_centers = torch.sum(loss_centers)
    loss_sigmas = torch.sum(loss_sigmas) 
    loss = loss_centers + loss_sigmas  

    return loss, loss_centers, loss_sigmas
