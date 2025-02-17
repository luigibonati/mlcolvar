
__all__ = ["GeneratorLoss", "compute_eigenfunctions","compute_covariance", "evaluate_eigenfunctions"]

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



    #gradient = torch.stack([torch.autograd.grad(outputs=output[:,idx].sum(), inputs=data, retain_graph=True, create_graph=True)[0] for idx in range(r)], dim=2).swapaxes(2,1) 
    compute_batch_jacobian = functorch.vmap(functorch.jacrev(self.model,argnums=0),in_dims=(0))
    gradient = compute_batch_jacobian(data.unsqueeze(1))
    gradient = gradient.reshape(weights.shape[0],output.shape[1],-1)

    
    if gradient_descriptors is None:
       gradient_positions = gradient * torch.sqrt(self.friction)
    else:
       gradient_positions = torch.einsum("ijk,imkl->ijml", gradient, gradient_descriptors) 
       gradient_positions = gradient_positions.reshape(-1,output.shape[1],gradient_descriptors.shape[1]*3)* torch.sqrt(self.friction) 
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


def compute_eigenfunctions(model, dataset, friction, eta, r, cell=None, tikhonov_reg=1e-4):
    """
    Computes eigenfunctions and eigenvalues from a learned representation.

    This function estimates the eigenfunctions and eigenvalues of the infinitesimal generator 
    associated with the Langevin process. The eigenvalues are computed using a resolvent approach, 
    where `evals` relate to the generator's eigenvalues as: `lambda = eta - 1/evals`.

    Parameters:
    ----------
    model : torch.nn.Module
        The neural network model used to compute the representation of input data.
    dataset : dict
        Dictionary containing:
            - 'data' (torch.Tensor, shape (N, d)): Input configurations.
            - 'weights' (torch.Tensor, shape (N,)): Probability weights associated with the data points.
    friction : torch.Tensor, shape (N,)
        Langevin friction values for each data point.
    eta : float
        Hyperparameter for the resolvent approach.
    r : int
        Number of eigenfunctions to compute.
    gradient_descriptors : torch.Tensor, optional, shape (N, d, M, 3)
        Derivatives of descriptors with respect to atomic positions. If `None`, 
        the function uses direct gradients of `model(X)`.
    cell : torch.Tensor, optional, shape (3,3)
        If provided, used to normalize the gradients when periodic boundary conditions apply.

    Returns:
    --------
    g : torch.Tensor, shape (N, r)
        The computed eigenfunctions evaluated at each data point.
    lambdas : torch.Tensor, shape (r,)
        The eigenvalues associated with the generator, sorted in descending order.
    evecs : torch.Tensor, shape (r, r)
        The eigenvectors of the operator.

    Notes:
    ------
    - Eigenfunctions are normalized using the dataset weights.
    - If `gradient_descriptors` is provided, the function projects the gradients onto it.
    - The operator matrix is regularized to improve numerical stability.
    """
    #friction=friction.to("cuda")
    dataset["data"].requires_grad = True
    X= dataset["data"]
    
    d=dataset["data"].shape[1]
    psi_X = model(X)
    gradient_X = torch.stack([torch.autograd.grad(outputs=psi_X[:,idx].sum(), inputs=X, retain_graph=True, create_graph=True)[0].reshape((-1,d)) for idx in range(r)], dim=2).swapaxes(2,1) 
    if "derivatives" in dataset.keys:
       gradient_positions = torch.einsum("ijk,imkl->ijml", gradient_X, dataset["derivatives"]) 
       gradient_positions = gradient_positions.reshape(-1,psi_X.shape[1],dataset["derivatives"].shape[1]*3)* torch.sqrt(friction)
       
    else:
       gradient_positions = gradient_X * torch.sqrt(friction)
    if cell is not None:
       gradient_positions /= cell
      
    weights_X = dataset["weights"]
    cov_X =  compute_covariance(psi_X, weights_X) 
    dcov_X =  compute_covariance(gradient_positions, weights_X) 

    W = eta *cov_X + dcov_X

    operator = torch.linalg.inv(W + tikhonov_reg*torch.eye(psi_X.size(1),device=psi_X.device))@cov_X
    evals, evecs = torch.linalg.eig(operator)
    g = psi_X @ evecs.real
    lambdas = eta - 1 / evals
    sorting = torch.argsort(-lambdas.real) 
    # Ensure normalization of eigenfunctions
    evecs.detach()[:,sorting] /= torch.sqrt(torch.mean(dataset["weights"].unsqueeze(1)*g**2,axis=0))
    g /= torch.sqrt(torch.mean(dataset["weights"].unsqueeze(1)*g**2,axis=0))
    return g[:,sorting], lambdas.detach()[sorting], evecs.detach()[:,sorting]

def evaluate_eigenfunctions(model, dataset, evecs):
    """
    Evaluates the eigenfunctions of a generator model.

    Parameters:
    -----------
    model : torch.nn.Module or callable
        A model that computes a representation of the input data.
    dataset : dict
        A dataset object.
    evecs : torch.Tensor
        A matrix of eigenvectors, that is the result from the function compute_eigenfunctions.
        If complex, only the real part is used.

    Returns:
    --------
    torch.Tensor
        The projected eigenfunctions, obtained by computing the dot product of the model's output 
        and the real part of `evecs`.
    """
    
    X = dataset["data"]
    psi_X = model(X)
    g = psi_X @ evecs.real
    return g

def forecast_state_occupation(eigenfunctions: torch.Tensor, 
                              eigenvalues: torch.Tensor,
                              times: torch.Tensor, 
                              classification: torch.Tensor,
                              weights: torch.Tensor, 
                              n_states: torch.Tensor):

    """
    Computes the time evolution of state occupation probabilities in a dynamical system.

    This function estimates the probability of being in a state, starting in another state 
    over time using eigenfunctions and eigenvalues of the system's generator.

    Parameters:
    -----------
    eigenfunctions : torch.Tensor, shape (N, r)
        The eigenfunctions evaluated at each sample point, where N is the number of samples 
        and r is the number of eigenfunctions.

    eigenvalues : torch.Tensor, shape (r,)
        The eigenvalues associated with the eigenfunctions.

    times : torch.Tensor, shape (n_times,)
        A 1D tensor containing the time points at which to evaluate the occupation probabilities.

    classification : torch.Tensor, shape (N,)
        A tensor assigning each sample point to a discrete state, with integer values in {0, ..., n_states-1}.

    weights : torch.Tensor, shape (N,)
        A weight associated with each sample point, used for proper normalization.

    n_states : int
        The total number of discrete states in the system.

    Returns:
    --------
    occupations : torch.Tensor, shape (n_states, n_states, n_times)
        A tensor where `occupations[i, j, t]` represents the probability of transitioning 
        from state `i` to state `j` at time `times[t]`.

    """
    
    # Number of samples
    N = classification.shape[0]

    # Create masks for each state
    state_masks = torch.arange(n_states, device=classification.device).view(-1, 1) == classification.unsqueeze(0)  # (n_states, N)

    # Compute initial state occupations
    inv_u_0 = (state_masks * weights).mean(dim=1, keepdim=True)  # (n_states, 1)
    u_0 = (state_masks / inv_u_0) 
    # Project onto eigenfunctions
    initial_state_on_basis = ((u_0 * weights) @ eigenfunctions) / N # (n_states, n_eigen)
    final_state_on_basis = ((state_masks * weights) @ eigenfunctions) / N  # Ensure proper mean normalization

    # Ensure eigenvalues are correctly shaped
    eigenvalues = eigenvalues.view(1, -1)  # (1, n_eigen)

    # Compute time evolution
    time_evolution = torch.exp(times.view(-1, 1) * eigenvalues)  # (n_times, n_eigen)

    # Compute occupation over time
    occupation_over_time = (
        (initial_state_on_basis[:, None, :] * final_state_on_basis[None, :, :])  # (n_states, n_states, n_eigen)
        @ time_evolution.T.real  # Matrix multiplication over n_eigen -> (n_states, n_states, n_times)
    )

    return occupation_over_time  # Shape: (n_states, n_states, n_times)