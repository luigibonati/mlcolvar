
__all__ = ["GeneratorLoss", "compute_eigenfunctions","compute_covariance"]

# =============================================================================
# GLOBAL IMPORTS
# =============================================================================
###TODO: write the loss as a function
###TODO: Try to use vmap + jacfwd instead of autograd or jacobian, it might be more efficient
###TODO: Maybe add tikohonov regularization as an hyperparameter

import torch
from torch_scatter import scatter
import functorch
class GeneratorLoss(torch.nn.Module):
  """
  Computes the loss to learn a representation for the generator
  """
  def __init__(self, model, eta, cell, friction, alpha, n_cvs):
    """
    model: nn.module
    Actually unused, this is stupid, it might be worse deleting it... 
    eta: float
    eta : float
      Hyperparameter for the shift to define the resolvent. $(\eta I-_mathcal{L})^{-1}$
    r : int
      Hyperparamer for the number of eigenfunctions wanted
    alpha : float
      Hyperparamer that scales the orthonormality loss
    cell : float
      CUBIC cell size length, used to scale the positions from reduce coordinates to real coordinates, by default None
    friction: torch.tensor 
      Langevin friction which should contain \sqrt{k_B*T/(gamma*m_i)}
    n_cvs:
      useless, remove it
    """
    super().__init__()
    self.model = model
    self.eta = eta
    self.friction = friction
    self.lambdas = torch.nn.Parameter(10*torch.randn(n_cvs), requires_grad=True)
    self.alpha = alpha
    self.cell= cell

  def compute_covariance(self,X,weights):
    """
    Computes covariance matrix of data. I think this function already exists in mlcolvar
    X: torch.tensor
      data for which we want to compute the covariance matrix
    weights: torch.tensor
      statistical weights
    """
    n = X.size(0)
    pre_factor = n / (n - 1)
    if X.ndim == 2:
        return   pre_factor * (torch.einsum("ij,ik,i->jk",X,X,weights)/n )#(X.T @ X / n - mean @ mean.T)
    else:
        return pre_factor * (torch.einsum("ijk,ilk,i->jl",X,X,weights) / n)

  def forward(self, data, output, weights, gradient_descriptors=None):
    """
    Computes the loss to learn the representation
    data: torch.tensor
      input of the NN
    output: torch.tensor
      output of the NN
    weights: torch.tensor
      Statistical weights
    gradient_descriptors: torch.tensor, optional
      Gradient of the descriptors with respect to the atomic positions.  
      Used for the chain rule to compute the full gradients
    """
    lambdas = self.lambdas**2
    diag_lamb = torch.diag(lambdas)
    #sorted_lambdas = lambdas[torch.argsort(lambdas)]
    r = output.shape[1]
    sample_size = output.shape[0]//2



    gradient = torch.stack([torch.autograd.grad(outputs=output[:,idx].sum(), inputs=data, retain_graph=True, create_graph=True)[0] for idx in range(r)], dim=2).swapaxes(2,1) 
    compute_batch_jacobian = functorch.vmap(functorch.jacrev(self.model,argnums=0),in_dims=(0))
    gradient_X = compute_batch_jacobian(data.unsqueeze(1))
    gradient = gradient.reshape(weights.shape[0],output.shape[1],-1)

    
    if gradient_descriptors is None:
       gradient_positions = gradient * self.friction
    else:
       gradient_positions = torch.einsum("ijk,imkl->ijml", gradient, gradient_descriptors) 
       gradient_positions = gradient_positions.reshape(-1,output.shape[1],gradient_descriptors.shape[1]*3)* self.friction 
    if self.cell is not None:
       gradient_positions /= (self.cell)
    weights_X, weights_Y = weights[:sample_size], weights[sample_size:]
    gradient_X, gradient_Y = gradient_positions[:sample_size], gradient_positions[sample_size:]
    psi_X, psi_Y = output[:sample_size], output[sample_size:]


    
    cov_X =  self.compute_covariance(psi_X , weights_X) 
    
    cov_Y =  self.compute_covariance(psi_Y , weights_Y)


    dcov_X =  self.compute_covariance(gradient_X , weights_X) 
 
    dcov_Y =  self.compute_covariance(gradient_Y , weights_Y) 
    
    W1 = (self.eta *cov_X + dcov_X ) @ diag_lamb
    W2 = (self.eta *cov_Y + dcov_Y) @ diag_lamb
    

    loss_ef = torch.trace( ((cov_X@diag_lamb) @ W2 + (cov_Y@diag_lamb)@W1)/2 - cov_X@diag_lamb - cov_Y@diag_lamb)

    # Compute loss_ortho
    loss_ortho = self.alpha * (torch.trace((torch.eye(output.shape[1], device=output.device) - cov_X).T @ (torch.eye(output.shape[1], device=output.device) - cov_X)))
    #loss_ortho = penalty
    loss = loss_ef + loss_ortho#loss_ortho
    return loss, loss_ef, loss_ortho
  


def compute_covariance(X,weights):
    n = X.size(0)
    pre_factor = 1.0
    if X.ndim == 2:
        return   pre_factor * (torch.einsum("ij,ik,i->jk",X,X,weights)/n)#(X.T @ X / n - mean @ mean.T)
    else:
        return pre_factor * (torch.einsum("ijk,ilk,i->jl",X,X,weights) / n)

def compute_eigenfunctions(model, dataset, friction, eta, r,gradient_descriptors=None, cell=None):
    """
    Computes eigenfunctions and eigenvalues given a representation learnt with model.
    outputs evals and evecs. Evals relates to the eigenvalues of the generator by eta - 1/evals. 
    The eigenfunctions can be computed by doing: output@evecs
    Parameters:
    ----------
    model: module.nn
      model to compute the representation
    dataset: DictDataset
      should contain 'data', 'weights' 
    friction: torch.tensor
      Langevin friction
    eta: float
      Hyperparameter for the computation of the resolvent
    r: int
      number of eigenfunctions
    gradient_descriptors: torch.tensor
      derivatives of descriptors with respect to atomic positions
    """
    #friction=friction.to("cuda")
    dataset["data"].requires_grad = True
    X= dataset["data"]
    d=dataset["data"].shape[1]
    psi_X = model(X)
    gradient_X = torch.stack([torch.autograd.grad(outputs=psi_X[:,idx].sum(), inputs=X, retain_graph=True, create_graph=True)[0].reshape((-1,d)) for idx in range(r)], dim=2).swapaxes(2,1) 
    if gradient_descriptors is None:
       gradient_positions = gradient_X * torch.sqrt(friction)
    else:
       gradient_positions = torch.einsum("ijk,imkl->ijml", gradient_X, gradient_descriptors) 
       gradient_positions = gradient_positions.reshape(-1,psi_X.shape[1],gradient_descriptors.shape[1]*3)* torch.sqrt(friction)
    if cell is not None:
       gradient_positions /= cell
      
    weights_X = dataset["weights"]
    cov_X =  compute_covariance(psi_X, weights_X) 
    dcov_X =  compute_covariance(gradient_positions, weights_X) 

    W = eta *cov_X + dcov_X

    operator = torch.linalg.inv(W + 1e-5*torch.eye(psi_X.size(1),device=psi_X.device))@cov_X
    evals, evecs = torch.linalg.eig(operator)
    return evals.detach(), evecs.detach()