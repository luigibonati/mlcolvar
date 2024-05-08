#!/usr/bin/env python

# =============================================================================
# MODULE DOCSTRING
# =============================================================================

"""
Target Discriminant Analysis Loss Function.
"""

__all__ = ["TDALoss", "tda_loss"]


# =============================================================================
# GLOBAL IMPORTS
# =============================================================================

from typing import Union, List, Tuple
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
        target_centers: Union[List[float], torch.Tensor],
        target_sigmas: Union[List[float], torch.Tensor],
        alpha: float = 1.0,
        beta: float = 100.0,
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
        alpha : float, optional
            Centers_loss component prefactor, by default 1.
        beta : float, optional
            Sigmas loss compontent prefactor, by default 100.
        """
        super().__init__()
        self.n_states = n_states
        self.target_centers = target_centers
        self.target_sigmas = target_sigmas
        self.alpha = alpha
        self.beta = beta

    def forward(
        self, H: torch.Tensor, labels: torch.Tensor, return_loss_terms: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Compute the value of the loss function.

        Parameters
        ----------
        H : torch.Tensor
            Shape ``(n_batches, n_features)``. Output of the NN.
        labels : torch.Tensor
            Shape ``(n_batches,)``. Labels of the dataset.
        return_loss_terms : bool, optional
            If ``True``, the loss terms associated to the center and standard
            deviations of the target Gaussians are returned as well. Default
            is ``False``.

        Returns
        -------
        loss : torch.Tensor
            Loss value.
        loss_centers : torch.Tensor, optional
            Only returned if ``return_loss_terms is True``. The value of the
            loss term associated to the centers of the target Gaussians.
        loss_sigmas : torch.Tensor, optional
            Only returned if ``return_loss_terms is True``. The value of the
            loss term associated to the standard deviations of the target Gaussians.
        """
        return tda_loss(
            H,
            labels,
            self.n_states,
            self.target_centers,
            self.target_sigmas,
            self.alpha,
            self.beta,
            return_loss_terms,
        )


def tda_loss(
    H: torch.Tensor,
    labels: torch.Tensor,
    n_states: int,
    target_centers: Union[List[float], torch.Tensor],
    target_sigmas: Union[List[float], torch.Tensor],
    alpha: float = 1,
    beta: float = 100,
    return_loss_terms: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
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
    alpha : float, optional
        Centers_loss component prefactor, by default 1.
    beta : float, optional
        Sigmas loss compontent prefactor, by default 100.
    return_loss_terms : bool, optional
        If ``True``, the loss terms associated to the center and standard deviations
        of the target Gaussians are returned as well. Default is ``False``.

    Returns
    -------
    loss : torch.Tensor
        Loss value.
    loss_centers : torch.Tensor, optional
        Only returned if ``return_loss_terms is True``. The value of the loss
        term associated to the centers of the target Gaussians.
    loss_sigmas : torch.Tensor, optional
        Only returned if ``return_loss_terms is True``. The value of the loss
        term associated to the standard deviations of the target Gaussians.
    """
    if not isinstance(target_centers, torch.Tensor):
        target_centers = torch.tensor(target_centers, dtype=H.dtype)
        target_centers.requires_grad_(True)
    if not isinstance(target_sigmas, torch.Tensor):
        target_sigmas = torch.tensor(target_sigmas, dtype=H.dtype)
        target_sigmas.requires_grad_(True)

    device = H.device
    target_centers = target_centers.to(device)
    target_sigmas = target_sigmas.to(device)
    loss_centers = torch.zeros_like(target_centers, device=device)
    loss_sigmas = torch.zeros_like(target_sigmas, device=device)

    for i in range(n_states):
        # check which elements belong to class i
        if not (labels == i).any():
            raise ValueError(
                f"State {i} was not represented in this batch! Either use bigger batch_size or a more equilibrated dataset composition!"
            )
        else:
            H_red = H[labels == i]

            # compute mean and standard deviation over the class i
            mu = torch.mean(H_red, 0)
            if len(torch.nonzero(labels == i)) == 1:
                warn(
                    f"There is only one sample for state {i} in this batch! Std is set to 0, this may affect the training! Either use bigger batch_size or a more equilibrated dataset composition!"
                )
                sigma = torch.tensor(0)
            else:
                sigma = torch.std(H_red, 0)

        # compute loss function contributes for class i
        loss_centers[i] = alpha * (mu - target_centers[i]).pow(2)
        loss_sigmas[i] = beta * (sigma - target_sigmas[i]).pow(2)

    # get total model loss
    loss_centers = torch.sum(loss_centers)
    loss_sigmas = torch.sum(loss_sigmas)
    loss = loss_centers + loss_sigmas

    if return_loss_terms:
        return loss, loss_centers, loss_sigmas
    return loss
