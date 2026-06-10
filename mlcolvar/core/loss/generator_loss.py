__all__ = ["GeneratorLoss"]

# =============================================================================
# GLOBAL IMPORTS
# =============================================================================

import torch
import torch_geometric
from typing import Union, Tuple
from mlcolvar.core.loss.utils.smart_derivatives import SmartDerivatives
from mlcolvar.utils._code import scatter_sum

class GeneratorLoss(torch.nn.Module):
    """
    Loss function used to learn a representation of the eigenspace of the
    infinitesimal generator.

    The loss jointly optimizes:

    1. A representation produced by the neural network.
    2. A set of trainable parameters ``lambdas`` that approximate the
       eigenvalues of the shifted resolvent operator.

    The objective combines:

    - a variational term enforcing consistency with the generator dynamics;
    - an orthonormality penalty on the learned representation.

    References
    ----------
    T. Devergne, V. Kostic, M. Pontil, M. Parrinello,
    "Slow dynamical modes from static averages",
    J. Chem. Phys., 2025.
    """

    def __init__(self,
                 r: int, 
                 eta: float, 
                 friction: torch.Tensor, 
                 alpha: float,
                 descriptors_derivatives: Union[SmartDerivatives, torch.Tensor] = None,
                 n_dim: int = 3,
                 split: bool = True,
                 softmax_postproc=True,
                 ):
        """
        Initialize the generator loss.

        Parameters
        ----------
        r : int
            Number of latent functions (network outputs) used to represent the
            generator eigenspace.

        eta : float
            Resolvent shift parameter defining the operator

            .. math::

                (\eta I - \mathcal{L})^{-1}

            where :math:`\mathcal{L}` is the infinitesimal generator.

        friction : torch.Tensor
            Langevin prefactor associated with each atom. The tensor is used to
            weight coordinate gradients when constructing the Dirichlet form.

        alpha : float
            Weight of the orthonormality regularization term:

            .. math::

                L = L_{\mathrm{var}} + \alpha L_{\mathrm{ortho}}

        descriptors_derivatives : SmartDerivatives or torch.Tensor, optional
            Descriptor derivatives with respect to atomic coordinates.

            Providing these derivatives avoids recomputing descriptor Jacobians
            during training.

            Supported formats are:

            - ``SmartDerivatives`` for memory-efficient sparse evaluation.
            - ``torch.Tensor`` containing the full descriptor Jacobian.

        n_dim : int, default=3
            Number of spatial dimensions.

        split : bool, default=True
            Whether to use the split-batch estimator when computing
            covariance matrices.

        softmax_postproc : bool, default=True
            Whether the model output has been augmented by the softmax
            post-processing layer. When enabled, a constant basis function is
            appended internally before constructing the loss.
        """
       
        super().__init__()
        self.eta = eta
        self.register_buffer("friction", friction)
        self.lambdas = torch.nn.Parameter(10 * torch.randn(r), requires_grad=True)
        self.alpha = alpha
        self.descriptors_derivatives = descriptors_derivatives
        self.n_dim = n_dim
        self.split=split
        self.softmax_postproc=softmax_postproc

    def forward(self,
                input : torch.Tensor,
                output : torch.Tensor, 
                weights : torch.Tensor,
                ref_idx : torch.Tensor = None
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        # preload descriptors matrix on device
        if isinstance(self.descriptors_derivatives, torch.Tensor):
            if self.descriptors_derivatives.device != input.device:
                self.descriptors_derivatives = self.descriptors_derivatives.to(input.device)

        return generator_loss(input=input,
                              output=output,
                              weights=weights,
                              eta=self.eta,
                              alpha=self.alpha,
                              friction=self.friction,
                              lambdas=self.lambdas,
                              descriptors_derivatives=self.descriptors_derivatives,
                              ref_idx=ref_idx,
                              n_dim=self.n_dim,
                              split=self.split,
                              softmax_postproc=self.softmax_postproc
                              )


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
                   descriptors_derivatives : Union[SmartDerivatives, torch.Tensor] = None,
                   ref_idx : torch.Tensor = None,
                   n_dim : int = 3,
                   split : bool = True,
                   softmax_postproc=True,
                   ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute the variational objective used to learn generator eigenfunctions.

    The neural-network outputs define a low-dimensional representation on which
    the shifted resolvent operator is approximated. The loss jointly optimizes
    the representation and a set of trainable eigenvalue parameters.

    The objective consists of:

    1. A variational term involving covariance matrices of the learned
    representation and its gradients.
    2. An orthonormality penalty that encourages independent eigenfunctions.

    When ``split=True``, unbiased covariance estimates are obtained using a
    two-way batch split


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
        Trainable parameters. After training, they should correspond to the resolvent eigenvalues.

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
    n_dim : int
        Number of dimensions, by default 3.
    split : bool
        Do we use split the batch to compute the loss

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        Total loss, eigenfunctions loss, orthonormality loss 
    """    
    if descriptors_derivatives is not None and ref_idx is None:
        raise ValueError ("Descriptors derivatives need reference indeces from the dataset! Use a dataset with the ref_idx, see docstrign for details")

    # ------------------------ SETUP ------------------------
    # get correct device
    

    # move and process lambdas to device
    
    if isinstance(input, torch_geometric.data.batch.Batch):
        _is_graph_data = True
        batch = torch.clone(input['batch'])
        node_types = torch.where(input['node_attrs'])[1]#last one corresponds to yes or no it is a secondary structure
        input = input['positions']
        device = input.device

    else:
        device = input.device
        _is_graph_data = False
    
    lambdas = lambdas.to(device)
    diag_lamb = torch.diag(lambdas**2)
    if softmax_postproc:
        diag_lamb = torch.block_diag(diag_lamb, torch.tensor(1.0,device=device).unsqueeze(0))
        one_column = torch.ones((output.shape[0],1),device=device)
        output = torch.cat((output,one_column),dim=1)
    # get number of outputs and sample sizes
    r = output.shape[1]
    sample_size = output.shape[0] // 2
    
    # expand friction tensor
    if _is_graph_data:
        friction = friction[node_types].unsqueeze(1).unsqueeze(2)
    else:
        friction = friction.repeat_interleave(n_dim) 
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
        gradient_positions = descriptors_derivatives(gradient, ref_idx).view(input.shape[0], -1, r)
    
    # --> If we directly pass the matrix d_desc/d_pos
    elif isinstance(descriptors_derivatives, torch.Tensor): 
        descriptors_derivatives = descriptors_derivatives.to(device)
        gradient_positions = torch.einsum("bdo,badx->baxo", gradient, descriptors_derivatives[ref_idx]).contiguous()
        gradient_positions = gradient_positions.view(input.shape[0],  # number of entries
                                                        descriptors_derivatives.shape[1] * 3, # number of atoms * 3 
                                                        output.shape[-1] # number of outputs
                                                        )
        
    # If the input was already positions
    else:
        gradient_positions = gradient 


    if r==1:
        gradient_positions = gradient_positions.unsqueeze(-1)

    # this is to make the following computation easier to write
    gradient_positions = gradient_positions.transpose(2,1).contiguous()

    try:
        gradient_positions = gradient_positions * torch.sqrt(friction)
    except RuntimeError as e:
        raise RuntimeError(e, """[HINT]: Is you system in 3 dimension? By default the code assumes so, if it's not the case change the n_dim key to the right dimensionality.""")
    if _is_graph_data:
        gradient_positions = torch.einsum("ikd,ild->ikl",gradient_positions, gradient_positions)
        gradient_positions = scatter_sum(gradient_positions, batch,dim=0)
    # ------------------------ COVARIANCES ------------------------
    if split:
        first = slice(0, sample_size)
        second = slice(sample_size, None)

        # In order to have unbiased estimation, we split the dataset in two chunks
        weights_X, weights_Y = weights[first], weights[second]
        #print(batch)
        gradient_X, gradient_Y = gradient_positions[first], gradient_positions[second]
        psi_X, psi_Y = output[first], output[second]
        #print(f"Weights X {weights_X.shape} \n Weights_Y {weights_Y.shape} \n psi_X {psi_X.shape} \n psi_Y {psi_Y.shape} \n {gradient_X.shape} \n {gradient_Y.shape} \n {weights[mask_var_batches_X]}")

        # compute covariances
        cov_X = compute_covariance(psi_X, weights_X)
        cov_Y = compute_covariance(psi_Y, weights_Y)
        if _is_graph_data:
            dcov_X=torch.einsum("ikl,i->kl",gradient_X, weights_X) / gradient_X.shape[0]
            dcov_Y=torch.einsum("ikl,i->kl",gradient_Y, weights_Y) / gradient_Y.shape[0]
        else:
            dcov_X = compute_covariance(gradient_X, weights_X)
            dcov_Y = compute_covariance(gradient_Y, weights_Y)

        # action of shifted generator on the two chunks
        W1 = ( cov_X + dcov_X / eta) 
        W2 = ( cov_Y + dcov_Y / eta) 



        loss_1 = 0.5*torch.einsum('i,ij,j,ji->',
                                  diag_lamb.diag(),W1,diag_lamb.diag(),W2)
        loss_1 += 0.5*torch.einsum('i,ij,j,ji->',
                                   diag_lamb.diag(),W2,diag_lamb.diag(),W1)
        
        loss_2 = - weights_Y.mean()*(torch.diagonal(cov_X) * diag_lamb).sum()
        loss_2 -= weights_X.mean()*(torch.diagonal(cov_Y) * diag_lamb).sum()
        
        #compute ortho-normality term || U^*U - I||_F^2 + || V^*V - I||_F^2, split the batch for uniased estiamtion
        loss_3 = torch.einsum('ij,ji->',
                            W1,W2)
        loss_3 -= torch.mean(weights_X)*torch.diagonal(W2).sum()
        loss_3 -= torch.mean(weights_Y)*torch.diagonal(W1).sum()
        loss_3 += psi_X.shape[1]*torch.mean(weights_X)*torch.mean(weights_Y)         

        # ------------------------ COMPUTE LOSSES ------------------------
    else:
        print("NOT IMPLEMENTED YET")
    # combine
    loss = (loss_1+loss_2+alpha*loss_3)

    return loss, (loss_1+loss_2).detach(), alpha*loss_3.detach()