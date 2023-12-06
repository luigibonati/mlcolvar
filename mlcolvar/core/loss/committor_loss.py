#!/usr/bin/env python

# =============================================================================
# MODULE DOCSTRING
# =============================================================================

"""
Committor function Loss Function.
"""

__all__ = ["CommittorLoss", "committor_loss"]

# =============================================================================
# GLOBAL IMPORTS
# =============================================================================

from typing import Union
from warnings import warn

import torch

# =============================================================================
# LOSS FUNCTIONS
# =============================================================================


class CommittorLoss(torch.nn.Module):
    """Compute a loss function based on Kolmogorov's variational principle for the determination of the committor function"""

    def __init__(self,
                mass: torch.Tensor,
                alpha : float,
                cell_size: float = None,
                gamma : float = 10000,
                delta_f: float = 0
                 ):
        """Compute Kolmogorov's variational principle loss and impose boundary condition on the metastable states

        Parameters
        ----------
        mass : torch.Tensor
            Atomic masses of the atoms in the system
        alpha : float
            Hyperparamer that scales the boundary conditions contribution to loss, i.e. alpha*(loss_bound_A + loss_bound_B)
        cell_size : float, optional
            CUBIC cell size length, used to scale the positions from reduce coordinates to real coordinates, by default None
        gamma : float, optional
            Hyperparamer that scales the whole loss to avoid too small numbers, i.e. gamma*(loss_var + loss_bound), by default 10000
        delta_f : float, optional
            Delta free energy between A (label 0) and B (label 1), units is kBT, by default 0. 
            State B is supposed to be higher in energy.
        """
        super().__init__()
        self.mass = mass
        self.alpha = alpha
        self.cell_size = cell_size
        self.gamma = gamma
        self.delta_f = delta_f

    def forward(
        self, x : torch.Tensor, q : torch.Tensor, labels : torch.Tensor, w : torch.Tensor, create_graph : bool = True
    ) -> torch.Tensor:
        return committor_loss(
            x,
            q,
            labels,
            w,
            self.mass,
            self.alpha,
            self.cell_size,
            self.gamma,
            self.delta_f,
            create_graph
        )


def committor_loss(x : torch.Tensor, 
                  q : torch.Tensor, 
                  labels: torch.Tensor, 
                  w: torch.Tensor,
                  mass: torch.Tensor,
                  alpha : float,
                  cell_size: float = None,
                  gamma : float = 10000,
                  delta_f: float = 0,
                  create_graph : bool = True):
    """Compute variational loss for committor optimization with boundary conditions

    Parameters
    ----------
    x : torch.Tensor
        Input of the NN
    q : torch.Tensor
        Committor quess q(x), it is the output of NN
    labels : torch.Tensor
        Labels for states, A and B states for boundary conditions
    w : torch.Tensor
        Reweighing factors. This depends on the simualtion in which the data were collected.
        It is standard reweighing: exp[-beta*V(x)]
    mass : torch.Tensor
        List of masses of all the atoms we are using, for each atom we need to repeat three times for x,y,z
    alpha : float
        Hyperparamer that scales the boundary conditions contribution to loss, i.e. alpha*(loss_bound_A + loss_bound_B)
    cell_size : float
        CUBIC cell size length, used to scale the positions from reduce coordinates to real coordinates, default None 
    gamma : float
        Hyperparamer that scales the whole loss to avoid too small numbers, i.e. gamma*(loss_var + loss_bound) 
        By default 10000
    delta_f : float
        Delta free energy between A (label 0) and B (label 1), units is kBT, by default 0. 
        State B is supposed to be higher in energy.
    create_graph : bool
        Make loss backwardable, deactivate for validation to save memory, default True
    
    Returns
    -------
    loss : torch.Tensor
        Loss value.
    gamma*loss_var : torch.Tensor
        The variational loss term
    gamma*alpha*loss_A : torch.Tensor
        The boundary loss term on basin A
    gamma*alpha*loss_B : torch.Tensor
        The boundary loss term on basin B
    """
    # inherit right device
    device = x.device 

    mass = mass.to(device)

    # Create masks to access different states data
    mask_A = torch.nonzero(labels.squeeze() == 0, as_tuple=True) 
    mask_B = torch.nonzero(labels.squeeze() == 1, as_tuple=True) 
    
    # Update weights of basin B using the information on the delta_f
    delta_f = torch.Tensor([delta_f])
    w[mask_B] = w[mask_B] * torch.exp(delta_f.to(device)) 

    ###### VARIATIONAL PRINICIPLE LOSS ######
    # Each loss contribution is scaled by the number of samples
    
    # We need the gradient of q(x)
    grad_outputs = torch.ones_like(q)
    grad = torch.autograd.grad(q, x, grad_outputs=grad_outputs, retain_graph=True, create_graph=create_graph)[0]

    # TODO this fixes cell size issue
    if cell_size is not None:
        grad = grad / cell_size

    # we sanitize the shapes of mass and weights tensors
    # mass should have size [1, n_atoms*spatial_dims]
    mass = mass.unsqueeze(0)
    # weights should have size [n_batch, 1]
    w = w.unsqueeze(-1)

    # we get the square of grad(q) and we multiply by the weight
    grad_square = torch.sum((torch.pow(grad, 2)*(1/mass)), axis=1, keepdim=True) * w

    # variational contribution to loss: we sum over the batch
    loss_var = torch.mean(grad_square)

    # boundary conditions
    q_A = q[mask_A]
    q_B = q[mask_B]
    loss_A = torch.mean( torch.pow(q_A, 2))
    loss_B = torch.mean( torch.pow( (q_B - 1) , 2))

    loss = gamma*( loss_var + alpha*(loss_A + loss_B) )
    
    # TODO maybe there is no need to detach them for logging
    return loss, gamma*loss_var.detach(), alpha*gamma*loss_A.detach(), alpha*gamma*loss_B.detach()