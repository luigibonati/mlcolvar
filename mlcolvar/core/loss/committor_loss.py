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
from typing import Tuple
from mlcolvar.core.loss.utils.smart_derivatives import SmartDerivatives

# =============================================================================
# LOSS FUNCTIONS
# =============================================================================


class CommittorLoss(torch.nn.Module):
    """Compute a loss function based on Kolmogorov's variational principle for the determination of the committor function"""

    def __init__(self,
                atomic_masses: torch.Tensor,
                alpha: float,
                cell: float = None,
                gamma: float = 10000.0,
                delta_f: float = 0.0,
                separate_boundary_dataset : bool = True,
                descriptors_derivatives : torch.nn.Module = None,
                log_var: bool = False,
                z_regularization: float = 0.0,
                n_dim : int = 3,
                 ):
        """Compute Kolmogorov's variational principle loss and impose boundary conditions on the metastable states

        Parameters
        ----------
        atomic_masses : torch.Tensor
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
        log_var : bool, optional
            Switch to minimize the log of the variational functional, by default False.
        z_regularization : float, optional
            Introduces a regularization on the learned z space avoiding too large absolute values.
            The magnitude of the regularization is scaled by the given number, by default 0.0
        n_dim : int
            Number of dimensions, by default 3.
        """
        super().__init__()
        self.register_buffer("atomic_masses", atomic_masses)
        self.alpha = alpha
        self.cell = cell
        self.gamma = gamma
        self.delta_f = delta_f
        self.descriptors_derivatives = descriptors_derivatives
        self.separate_boundary_dataset = separate_boundary_dataset
        self.log_var = log_var
        self.z_regularization = z_regularization

    def forward(self, 
                x: torch.Tensor,
                z: torch.Tensor,
                q: torch.Tensor, 
                labels: torch.Tensor, 
                w: torch.Tensor, 
                ref_idx: torch.Tensor = None, 
                create_graph: bool = True
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Committor loss forward pass

        Parameters
        ----------
        x : torch.Tensor
            Model input, i.e., either positions or descriptors if using descriptors_derivatives
        z : torch.Tensor
            Model unactivated output, i.e., z value
        q : torch.Tensor
            Model final output, i.e., committor value
        labels : torch.Tensor
            Input labels
        w : torch.Tensor
            Input weights
        ref_idx : torch.Tensor, optional
            Reference indeces for the unshuffled dataset for properly handling batching/splitting/shuffling
            when descriptors derivatives are provided, by default None. 
            Ref_idx can be generated automatically using SmartDerivatives or by setting create_ref_idx=True when initializing a DictDataset.
            See also mlcolvar.core.loss.utils.smart_derivatives.SmartDerivatives
        create_graph : bool, optional
            Whether to create the graph during the computation for backpropagation, by default True

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
            Total loss and its components, i.e., variational, boundary A, and boundary B
        """
        return committor_loss(x=x,
                              z=z,
                              q=q,
                              labels=labels,
                              w=w,
                              atomic_masses=self.atomic_masses,
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
                   z: torch.Tensor,
                   q: torch.Tensor, 
                   labels: torch.Tensor, 
                   w: torch.Tensor,
                   atomic_masses: torch.Tensor,
                   alpha: float,
                   gamma: float = 10000,
                   delta_f: float = 0,
                   create_graph: bool = True,
                   cell: float = None,
                   separate_boundary_dataset: bool = True,
                   descriptors_derivatives: torch.nn.Module = None,
                   log_var: bool = False,
                   z_regularization: float = 0.0,
                   ref_idx: torch.Tensor = None,
                   n_dim : int = 3,
                  ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute variational loss for committor optimization with boundary conditions

    Parameters
    ----------
    x : torch.Tensor
        Input of the NN
    z : torch.Tensor
        z value z(x), it is the unactivated output of NN
    q : torch.Tensor
        Committor guess q(x), it is the output of NN
    labels : torch.Tensor
        Labels for states, A and B states for boundary conditions
    w : torch.Tensor
        Reweighing factors to Boltzmann distribution. This should depend on the simulation in which the data were collected.
    atomic_masses : torch.Tensor
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
    log_var : bool, optional
        Switch to minimize the log of the variational functional, by default False.
    z_regularization : float, optional
        Introduces a regularization on the learned z space avoiding too large absolute values.
        The magnitude of the regularization is scaled by the given number, by default 0.0
    ref_idx: torch.Tensor, optional
        Reference indeces for the unshuffled dataset for properly handling batching/splitting/shuffling
        when descriptors derivatives are provided, by default None. 
        Ref_idx can be generated automatically using SmartDerivatives or by setting create_ref_idx=True when initializing a DictDataset.
        See also mlcolvar.core.loss.utils.smart_derivatives.SmartDerivatives
    n_dim : int
        Number of dimensions, by default 3.

    Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
            Total loss and its components, i.e., variational, boundary A, and boundary B
    """
    if descriptors_derivatives is not None and ref_idx is None:
        raise ValueError ("Descriptors derivatives need reference indeces from the dataset! Use a dataset with the ref_idx, see docstrign for details")

    # ------------------------ SETUP ------------------------
    # inherit right device
    device = x.device 

    # expand mass tensor to [1, n_atoms*spatial_dims]
    atomic_masses = atomic_masses.to(device)
    atomic_masses = atomic_masses.repeat_interleave(n_dim) 

    # squeeze labels
    labels = labels.squeeze()


    # Create masks to access different states data
    mask_A = torch.nonzero(labels == 0, as_tuple=True) 
    mask_B = torch.nonzero(labels == 1, as_tuple=True)

    # create mask for variational data
    if separate_boundary_dataset:
        mask_var = torch.nonzero(labels > 1, as_tuple=True) 
    else: 
        mask_var = torch.ones(len(labels), dtype=torch.bool)


    # Update weights of basin B using the information on the delta_f
    delta_f = torch.Tensor([delta_f])
    # B higher in energy --> A-B < 0
    if delta_f < 0: 
        w[mask_B] = w[mask_B] * torch.exp(delta_f.to(device))
    # A higher in energy --> A-B > 0
    elif delta_f > 0:
        w[mask_A] = w[mask_A] * torch.exp(-delta_f.to(device)) 

    # weights should have size [n_batch, 1]
    w = w.unsqueeze(-1)


    # ------------------------  LOSS ------------------------
    # Each loss contribution is scaled by the number of samples
    
    # 1. ----- VARIATIONAL LOSS
    # Compute gradients of q(x) wrt x
    grad_outputs = torch.ones_like(q[mask_var])
    grad = torch.autograd.grad(q[mask_var], 
                               x, 
                               grad_outputs=grad_outputs, 
                               retain_graph=True, 
                               create_graph=create_graph)[0]
    grad = grad[mask_var]
    
    if cell is not None:
        grad = grad / cell
    
    # in case the input is not positions but descriptors, we need to correct the gradients up to the positions
    if isinstance(descriptors_derivatives, SmartDerivatives):
        # we use the precomputed derivatives from descriptors to pos
        gradient_positions = descriptors_derivatives(grad, ref_idx[mask_var]).view(x[mask_var].shape[0], -1)
    
    # If the input was already positions
    else:
        gradient_positions = grad

    # we do the square
    grad_square = torch.pow(gradient_positions, 2)
    
    # multiply by masses
    try:
        grad_square = torch.sum((grad_square * (1/atomic_masses)), 
                                 axis=1, 
                                 keepdim=True)    
    except RuntimeError as e:
        raise RuntimeError(e, """[HINT]: Is you system in 3 dimension? By default the code assumes so, if it's not the case change the n_dim key to the right dimensionality.""")

    # multiply by weights
    grad_square = grad_square * w[mask_var]

    # variational contribution to loss: we sum over the batch
    loss_var = torch.mean(grad_square)
    if log_var:
        loss_var = torch.log(loss_var + 1)
    else:
        loss_var = gamma*loss_var


    # 2. ----- BOUNDARY LOSS
    loss_A = gamma * torch.mean( torch.pow(q[mask_A], 2))
    loss_B = gamma * torch.mean( torch.pow( (q[mask_B] - 1) , 2))


    # 3. ----- OPTIONAL regularization on z
    if z_regularization != 0.0:
        loss_z_diff = z_regularization * (z.mean().abs() - z.mean().abs()).pow(2)
    else:
        loss_z_diff = 0
   
   
    # 4. ----- TOTAL LOSS
    loss = loss_var + alpha*(loss_A + loss_B) + loss_z_diff
    
    return loss, loss_var.detach(), alpha*loss_A.detach(), alpha*loss_B.detach()