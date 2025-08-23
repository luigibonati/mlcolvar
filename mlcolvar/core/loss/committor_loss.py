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
from typing import Tuple, Union
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
                descriptors_derivatives : Union[SmartDerivatives, torch.Tensor] = None,
                log_var: bool = False,
                z_regularization: float = 0.0,
                z_threshold: float = None,
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
        descriptors_derivatives : Union[SmartDerivatives, torch.Tensor], optional
            Derivatives of descriptors wrt atomic positions (if used) to speed up calculation of gradients, by default None. 
            Can be either:
                - A `SmartDerivatives` object to save both memory and time, see also mlcolvar.core.loss.committor_loss.SmartDerivatives
                - A torch.Tensor with the derivatives to save time, memory-wise could be less efficient
        ref_idx: torch.Tensor, optional
            Reference indeces for the unshuffled dataset for properly handling batching/splitting/shuffling
            when descriptors derivatives are provided, by default None. 
            Ref_idx can be generated automatically using SmartDerivatives or by setting create_ref_idx=True when initializing a DictDataset.
            See also mlcolvar.core.loss.utils.smart_derivatives.SmartDerivatives
        log_var : bool, optional
            Switch to minimize the log of the variational functional, by default False.
        z_regularization : float, optional
            Scales a regularization on the learned z space preventing it from exceeding the threshold given with 'z_threshold'.
            The magnitude of the regularization is scaled by the given number, by default 0.0
        z_threshold : float, optional
            Sets a maximum threshold for the z value during the training, by default None. 
            The magnitude of the regularization term is scaled via the `z_regularization` key.
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
        self.z_threshold = z_threshold
        self.n_dim = n_dim

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
                              z_regularization=self.z_regularization,
                              z_threshold=self.z_threshold,
                              ref_idx=ref_idx,
                              n_dim=self.n_dim
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
                   descriptors_derivatives: Union[SmartDerivatives, torch.Tensor] = None,
                   log_var: bool = False,
                   z_regularization: float = 0.0,
                   z_threshold : float = None,
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
    descriptors_derivatives : Union[SmartDerivatives, torch.Tensor], optional
        Derivatives of descriptors wrt atomic positions (if used) to speed up calculation of gradients, by default None. 
        Can be either:
            - A `SmartDerivatives` object to save both memory and time, see also mlcolvar.core.loss.committor_loss.SmartDerivatives
            - A torch.Tensor with the derivatives to save time, memory-wise could be less efficient
    log_var : bool, optional
        Switch to minimize the log of the variational functional, by default False.
    z_regularization : float, optional
        Scales a regularization on the learned z space preventing it from exceeding the threshold given with 'z_threshold'.
        The magnitude of the regularization is scaled by the given number, by default 0.0
    z_threshold : float, optional
        Sets a maximum threshold for the z value during the training, by default None. 
        The magnitude of the regularization term is scaled via the `z_regularization` key.
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

    if isinstance(descriptors_derivatives, torch.Tensor) and separate_boundary_dataset:
        raise ValueError ("Descriptors derivatives via explicit tensor are not implemented with separate_boundary_dataset key! Either use SmartDerivatives or deactivate separate_boundary_dataset")
    
    if (z_threshold is not None and (z_regularization == 0 or z_threshold <= 0)) or (z_threshold is None and z_regularization != 0) or z_regularization < 0:
        raise ValueError(f"To apply the regularization on z space both z_threshold and z_regularization key must be positive. Found {z_threshold} and {z_regularization}!")

    # ------------------------ SETUP ------------------------
    # inherit right device
    device = x.device 

    # expand mass tensor to [1, n_atoms*spatial_dims]
    atomic_masses = atomic_masses.to(device).repeat_interleave(n_dim) 

    # squeeze labels
    labels = labels.squeeze()


    # Create masks to access different states data
    mask_A = labels == 0
    mask_B = labels == 1

    # create mask for variational data
    if separate_boundary_dataset:
        mask_var = labels > 1
    else: 
        mask_var = torch.ones_like(labels, dtype=torch.bool)


    # Update weights of basin B using the information on the delta_f
    delta_f = torch.Tensor([delta_f]).to(device)
    # B higher in energy --> A-B < 0
    if delta_f < 0: 
        w[mask_B] *= torch.exp(delta_f)
    # A higher in energy --> A-B > 0
    elif delta_f > 0:
        w[mask_A] *= torch.exp(-delta_f) 

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
    
    # --> If we directly pass the matrix d_desc/d_pos
    elif isinstance(descriptors_derivatives, torch.Tensor): 
        descriptors_derivatives = descriptors_derivatives.to(device)
        gradient_positions = torch.einsum("bd,badx->bax", grad, descriptors_derivatives[ref_idx[mask_var]]).contiguous()
        gradient_positions = gradient_positions.view(x[mask_var].shape[0], -1)
    
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

    # variational contribution to loss: we sum over the batch
    loss_var = torch.mean(grad_square * w[mask_var])
    if log_var:
        loss_var = torch.log1p(loss_var)
    else:
        loss_var *= gamma


    # 2. ----- BOUNDARY LOSS
    loss_A = gamma * torch.mean( q[mask_A].pow(2) )
    loss_B = gamma * torch.mean( (q[mask_B] - 1).pow(2) )


    # 3. ----- OPTIONAL regularization on z
    if z_threshold is not None:
        over_threshold = torch.relu(z.abs() - z_threshold)
        loss_z_diff = z_regularization  * torch.mean(over_threshold.pow(2))
    else:
        loss_z_diff = 0

    # 4. ----- TOTAL LOSS
    loss = loss_var + alpha*(loss_A + loss_B) + loss_z_diff
    
    return loss, loss_var.detach(), alpha*loss_A.detach(), alpha*loss_B.detach()