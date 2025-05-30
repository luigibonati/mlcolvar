__all__ = ["GeneratorLoss"]

# =============================================================================
# GLOBAL IMPORTS
# =============================================================================

import torch
from typing import Union, Tuple
from mlcolvar.core.loss.committor_loss import SmartDerivatives


class GeneratorLoss(torch.nn.Module):
    """Computes the loss function to learn a representation for the infinitesimal generator"""

    def __init__(self,
                 r : int, 
                 eta : float, 
                 friction : torch.Tensor, 
                 alpha : float,
                 cell : float = None,  
                 descriptors_derivatives : Union[SmartDerivatives, torch.Tensor] = None
                 ):
        """TODO _summary_

        Parameters
        ----------
        r : int
            Number of eigenfunctions wanted, i.e., number of outputs of model.
        eta : float
            Hyperparameter for the shift to define the resolvent, i.e., $(\eta I-_mathcal{L})^{-1}$
        friction : torch.Tensor
            Langevin friction, i.e., $\sqrt{k_B*T/(gamma*m_i)}$
        alpha : float
            Hyperparamer that scales the contribution of orthonormality loss to the total loss, i.e., L = L_ef + alpha*L_ortho
        cell : float, optional
            CUBIC cell size length, used to scale the positions from reduce coordinates to real coordinates, by default None 
        descriptors_derivatives : Union[SmartDerivatives, torch.Tensor], optional
            Derivatives of descriptors wrt atomic positions (if used) to speed up calculation of gradients, by default None. 
            Can be either:
                - A `SmartDerivatives` object to save both memory and time, see also mlcolvar.core.loss.committor_loss.SmartDerivatives
                - A torch.Tensor with the derivatives to save time, memory-wise could be less efficient
        """
        super().__init__()

        self.eta = eta
        self.register_buffer("friction", friction)
        self.lambdas = torch.nn.Parameter(10 * torch.randn(r), requires_grad=True)
        self.alpha = alpha
        self.cell = cell
        self.descriptors_derivatives = descriptors_derivatives

    def forward(self,
                input : torch.Tensor,
                output : torch.Tensor, 
                weights : torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        return generator_loss(input=input,
                              output=output,
                              weights=weights,
                              eta=self.eta,
                              alpha=self.alpha,
                              friction=self.friction,
                              lambdas=self.lambdas,
                              cell=self.cell,
                              descriptors_derivatives=self.descriptors_derivatives
                              )


# TODO check that maybe we can replace this by the one from deepTICA
def compute_covariance(X, weights):
    n = X.size(0)
    pre_factor = n / (n - 1)
    if X.ndim == 2:
        return pre_factor * (
            torch.einsum("ij,ik,i->jk", X, X, weights) / n
        )  # (X.T @ X / n - mean @ mean.T)
    else:
        return pre_factor * (torch.einsum("ijk,ilk,i->jl", X, X, weights) / n)


def generator_loss(input : torch.Tensor,
                   output : torch.Tensor,
                   weights : torch.Tensor,
                   eta : float,
                   alpha : float,
                   friction : torch.Tensor,
                   lambdas : torch.Tensor,
                   cell : float = None,
                   descriptors_derivatives : Union[SmartDerivatives, torch.Tensor] = None
                   ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """TODO _summary_

    Parameters
    ----------
    input : torch.Tensor
        Input of the (set of) neural networks
    output : torch.Tensor
        Output of the (set of) neural networks
    weights : torch.Tensor
        Statistical weights of the samples, this could be from reweighting.
    eta : float
        Hyperparameter for the shift to define the resolvent, i.e., $(\eta I-_mathcal{L})^{-1}$
    alpha : float
        Hyperparamer that scales the contribution of orthonormality loss to the total loss, i.e., L = L_ef + alpha*L_ortho
    friction : torch.Tensor
        Langevin friction, i.e., $\sqrt{k_B*T/(gamma*m_i)}$
    lambdas : torch.Tensor
        TODO
    cell : float, optional
        CUBIC cell size length, used to scale the positions from reduce coordinates to real coordinates, by default None
    descriptors_derivatives : Union[SmartDerivatives, torch.Tensor], optional
        Derivatives of descriptors wrt atomic positions (if used) to speed up calculation of gradients, by default None. 
        Can be either:
            - A `SmartDerivatives` object to save both memory and time, see also mlcolvar.core.loss.committor_loss.SmartDerivatives
            - A torch.Tensor with the derivatives to save time, memory-wise could be less efficient

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        Total loss, eigenfunctions loss, orthonormality loss 
    """
    # ------------------------ SETUP ------------------------
    # get correct device
    device = input.device

    # move and process lambdas to device
    lambdas = lambdas.to(device)
    diag_lamb = torch.diag(lambdas**2)
    
    # get number of outputs and sample sizes
    r = output.shape[1]
    sample_size = output.shape[0] // 2
    
    
    # ------------------------ GRADIENTS ------------------------    
    # compute gradients of output wrt to the input iterating on the outputs
    grad_outputs = torch.ones(len(output), device=device)
    gradient = torch.stack([torch.autograd.grad(outputs=output[:, idx],
                                                inputs=input,
                                                grad_outputs=grad_outputs, 
                                                retain_graph=True, 
                                                create_graph=True)[0] for idx in range(r)
                            ], dim=2)
    
    
    # in case the input is not positions but descriptors, we need to correct the gradients up to the positions
    # --> If we pass a SmartDerivative object that takes the nonzero elements of the matrix d_desc/d_pos
    if isinstance(descriptors_derivatives, SmartDerivatives):
        gradient_positions = descriptors_derivatives(gradient).reshape(input.shape[0], -1, r)
    
    # --> If we directly pass the matrix d_desc/d_pos
    elif isinstance(descriptors_derivatives, torch.Tensor): 
        descriptors_derivatives = descriptors_derivatives.to(device)
        gradient_positions = torch.einsum("bdo,badx->baxo", gradient, descriptors_derivatives)
        gradient_positions = gradient_positions.reshape(input.shape[0],  # number of entries
                                                        descriptors_derivatives.shape[1] * 3, # number of atoms * 3 
                                                        output.shape[-1] # number of outputs
                                                        )
        
    # If the input was already positions
    else:
        gradient_positions = gradient 

    if cell is not None:
        gradient_positions /= cell

    if r==1:
        gradient_positions = gradient_positions.unsqueeze(-1)

    gradient_positions = gradient_positions.swapaxes(2,1)
    
    # multiply by friction
    # TODO change to have a simpler mass tensor
    gradient_positions = gradient_positions * torch.sqrt(friction)


    # ------------------------ COVARIANCES ------------------------

    # In order to have unbiased estimation, we split the dataset in two chunks TODO what?
    weights_X, weights_Y = weights[:sample_size], weights[sample_size:]
    gradient_X, gradient_Y = gradient_positions[:sample_size], gradient_positions[sample_size:]
    psi_X, psi_Y = output[:sample_size], output[sample_size:]

    # compute covariances
    cov_X = compute_covariance(psi_X, weights_X)
    cov_Y = compute_covariance(psi_Y, weights_Y)
    dcov_X = compute_covariance(gradient_X, weights_X)
    dcov_Y = compute_covariance(gradient_Y, weights_Y)

    # TODO what are these?
    W1 = (eta * cov_X + dcov_X) @ diag_lamb
    W2 = (eta * cov_Y + dcov_Y) @ diag_lamb


    # ------------------------ COMPUTE LOSSES ------------------------

    # Unbiased estimation of the "variational part"
    # It might be worth replacing with einsum if it is faster
    loss_ef = torch.trace(
        ((cov_X @ diag_lamb) @ W2 + (cov_Y @ diag_lamb) @ W1) / 2
        - cov_X @ diag_lamb
        - cov_Y @ diag_lamb
    )

    # Orthonormality part
    loss_ortho = alpha * (
        torch.trace(
            (torch.eye(output.shape[1], device=output.device) - cov_X).T
            @ (torch.eye(output.shape[1], device=output.device) - cov_Y)
        )
    )

    # combine
    loss = loss_ef + loss_ortho

    return loss, loss_ef.detach(), loss_ortho.detach()