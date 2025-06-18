#!/usr/bin/env python

# =============================================================================
# MODULE DOCSTRING
# =============================================================================

"""
Committor function Loss Function
"""

__all__ = ["CommittorLoss", "committor_loss"]

# =============================================================================
# GLOBAL IMPORTS
# =============================================================================

import torch

# =============================================================================
# LOSS FUNCTIONS
# =============================================================================


class CommittorLoss(torch.nn.Module):
    """Compute a loss function based on Kolmogorov's variational principle for the determination of the committor function"""

    def __init__(self,
                mass: torch.Tensor,
                alpha: float,
                cell: float = None,
                gamma: float = 10000,
                delta_f: float = 0,
                separate_boundary_dataset : bool = True,
                descriptors_derivatives : torch.nn.Module = None
                 ):
        """Compute Kolmogorov's variational principle loss and impose boundary conditions on the metastable states

        Parameters
        ----------
        mass : torch.Tensor
            Atomic masses of the atoms in the system
        alpha : float
            Hyperparamer that scales the boundary conditions contribution to loss, i.e. alpha*(loss_bound_A + loss_bound_B)
        cell : float, optional
            CUBIC cell size length, used to scale the positions from reduce coordinates to real coordinates, by default None
        gamma : float, optional
            Hyperparamer that scales the whole loss to avoid too small numbers, i.e. gamma*(loss_var + loss_bound), by default 10000
        delta_f : float, optional
            Delta free energy between A (label 0) and B (label 1), units is kBT, by default 0. 
            State B is supposed to be higher in energy.
        separate_boundary_dataset : bool, optional
            Switch to exculde boundary condition labeled data from the variational loss, by default True
        descriptors_derivatives : torch.nn.Module, optional
            `SmartDerivatives` object to save memory and time when using descriptors.
            See also mlcolvar.core.loss.committor_loss.SmartDerivatives
        ref_idx: torch.Tensor, optional
            Reference indeces for the unshuffled dataset for properly handling batching/splitting/shuffling
            when descriptors derivatives are provided, by default None. 
            Ref_idx can be generated automatically using SmartDerivatives or by setting create_ref_idx=True when initializing a DictDataset.
            See also mlcolvar.core.loss.utils.smart_derivatives.SmartDerivatives

        """
        super().__init__()
        self.register_buffer("mass", mass)
        self.alpha = alpha
        self.cell = cell
        self.gamma = gamma
        self.delta_f = delta_f
        self.descriptors_derivatives = descriptors_derivatives
        self.separate_boundary_dataset = separate_boundary_dataset

    def forward(
        self, x: torch.Tensor, q: torch.Tensor, labels: torch.Tensor, w: torch.Tensor, ref_idx: torch.Tensor = None, create_graph: bool = True
    ) -> torch.Tensor:
        return committor_loss(x=x,
                                q=q,
                                labels=labels,
                                w=w,
                                mass=self.mass,
                                alpha=self.alpha,
                                gamma=self.gamma,
                                delta_f=self.delta_f,
                                create_graph=create_graph,
                                cell=self.cell,
                                separate_boundary_dataset=self.separate_boundary_dataset,
                                descriptors_derivatives=self.descriptors_derivatives,
                                ref_idx=ref_idx,
                            )


def committor_loss(x: torch.Tensor, 
                  q: torch.Tensor, 
                  labels: torch.Tensor, 
                  w: torch.Tensor,
                  mass: torch.Tensor,
                  alpha: float,
                  gamma: float = 10000,
                  delta_f: float = 0,
                  create_graph: bool = True,
                  cell: float = None,
                  separate_boundary_dataset : bool = True,
                  descriptors_derivatives : torch.nn.Module = None,
                  ref_idx: torch.Tensor = None,
                  ):
    """Compute variational loss for committor optimization with boundary conditions

    Parameters
    ----------
    x : torch.Tensor
        Input of the NN
    q : torch.Tensor
        Committor guess q(x), it is the output of NN
    labels : torch.Tensor
        Labels for states, A and B states for boundary conditions
    w : torch.Tensor
        Reweighing factors to Boltzmann distribution. This should depend on the simulation in which the data were collected.
    mass : torch.Tensor
        List of masses of all the atoms we are using, for each atom we need to repeat three times for x,y,z.
        Can be created using `committor.utils.initialize_committor_masses`
    alpha : float
        Hyperparamer that scales the boundary conditions contribution to loss, i.e. alpha*(loss_bound_A + loss_bound_B)
    gamma : float
        Hyperparamer that scales the whole loss to avoid too small numbers, i.e. gamma*(loss_var + loss_bound) 
        By default 10000
    delta_f : float
        Delta free energy between A (label 0) and B (label 1), units is kBT, by default 0. 
    create_graph : bool
        Make loss backwardable, deactivate for validation to save memory, default True
    cell : float
        CUBIC cell size length, used to scale the positions from reduce coordinates to real coordinates, default None 
    separate_boundary_dataset : bool, optional
            Switch to exculde boundary condition labeled data from the variational loss, by default True
    descriptors_derivatives : torch.nn.Module, optional
        `SmartDerivatives` object to save memory and time when using descriptors.
        See also mlcolvar.core.loss.utils.smart_derivatives.SmartDerivatives
    ref_idx: torch.Tensor, optional
        Reference indeces for the unshuffled dataset for properly handling batching/splitting/shuffling
        when descriptors derivatives are provided, by default None. 
        Ref_idx can be generated automatically using SmartDerivatives or by setting create_ref_idx=True when initializing a DictDataset.
        See also mlcolvar.core.loss.utils.smart_derivatives.SmartDerivatives

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
    if descriptors_derivatives is not None and ref_idx is None:
        raise ValueError ("Descriptors derivatives need reference indeces from the dataset! Use a dataset with the ref_idx, see docstrign for details")

    # inherit right device
    device = x.device 

    mass = mass.to(device)

    labels = labels.squeeze()

    # Create masks to access different states data
    mask_A = torch.nonzero(labels == 0, as_tuple=True) 
    mask_B = torch.nonzero(labels == 1, as_tuple=True)
    if separate_boundary_dataset:
        mask_var = torch.nonzero(labels > 1, as_tuple=True) 
    else: 
        mask_var = torch.ones(len(x), dtype=torch.bool)

    if separate_boundary_dataset:
        mask_var = labels > 1
    else:
        mask_var = torch.ones_like(labels, dtype=torch.bool)
    
    # Update weights of basin B using the information on the delta_f
    delta_f = torch.Tensor([delta_f])
    if delta_f < 0: # B higher in energy --> A-B < 0
        w[mask_B] = w[mask_B] * torch.exp(delta_f.to(device))
    elif delta_f > 0: # A higher in energy --> A-B > 0
        w[mask_A] = w[mask_A] * torch.exp(-delta_f.to(device)) 

    ###### VARIATIONAL PRINICIPLE LOSS ######
    # Each loss contribution is scaled by the number of samples
    
    # We need the gradient of q(x)
    grad_outputs = torch.ones_like(q[mask_var])
    grad = torch.autograd.grad(q[mask_var], x, grad_outputs=grad_outputs, retain_graph=True, create_graph=create_graph)[0]
    grad = grad[mask_var]
    
    # TODO this fixes cell size issue
    if cell is not None:
        grad = grad / cell
    
    if descriptors_derivatives is not None:
        grad = descriptors_derivatives(grad, ref_idx[mask_var]).reshape(x[mask_var].shape[0], -1)
    
    # we do the square
    grad_square = torch.pow(grad, 2)
        
    # we sanitize the shapes of mass and weights tensors
    # mass should have size [1, n_atoms*spatial_dims]
    # TODO change to have a simpler mass tensor
    mass = mass.unsqueeze(0)
    # weights should have size [n_batch, 1]
    w = w.unsqueeze(-1)

    grad_square = torch.sum((grad_square * (1/mass)), axis=1, keepdim=True)    
    grad_square = grad_square * w[mask_var]

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
