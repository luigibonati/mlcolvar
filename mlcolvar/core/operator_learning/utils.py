import torch
import lightning
import torch_geometric
from typing import Union, Tuple
from mlcolvar.core import FeedForward
from mlcolvar.core.loss.utils.smart_derivatives import SmartDerivatives
from mlcolvar.data import DictDataset
from mlcolvar.utils._code import scatter_sum
import numpy as np

from mlcolvar.data import DictDataset
import gc

class SoftmaxPostProcessing(torch.nn.Module):
    """Apply a softmax normalization followed by a learnable linear mixing.

    Parameters
    ----------
    r : int, default=4
        Number of representation channels. This is both the input and output
        dimension of the final linear layer.
    """

    def __init__(self, r: int = 4):
        super().__init__()
        self.final_linear = torch.nn.Linear(r, r)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Normalize the last dimension and apply the final linear layer.

        Parameters
        ----------
        input : torch.Tensor
            Tensor of shape ``(..., r)``.

        Returns
        -------
        torch.Tensor
            Tensor of shape ``(..., r)`` after softmax normalization and
            linear projection.
        """
        input = torch.nn.functional.softmax(input, dim=-1)
        return self.final_linear(input)

def sqrtmh(A: torch.Tensor):
    L, Q = torch.linalg.eigh(A)
    zero = torch.zeros((), device=L.device, dtype=L.dtype)
    threshold = L.max(-1).values * L.size(-1) * torch.finfo(L.dtype).eps
    L = L.where(L > threshold.unsqueeze(-1), zero)  # zero out small components
    return (Q * L.sqrt().unsqueeze(-2)) @ Q.mH

def compute_covariances(input,output,weights,r,friction,n_dim=3,descriptors_derivatives : Union[SmartDerivatives, torch.Tensor] = None, ref_idx=None):
    """Compute covariance of features and covariance of gradient matrices.

    The function evaluates two weighted matrices:

    - the covariance of the learned representation;
    - the covariance of its gradients with respect to atomic positions.

    If the model input is a descriptor rather than atomic coordinates,
    ``descriptors_derivatives`` is used to map descriptor gradients back to
    position gradients.

    Parameters
    ----------
    input : torch.Tensor or torch_geometric.data.Batch
        Model input. Can be atomic positions, descriptors, or graph data.
    output : torch.Tensor
        Model output with shape ``(n_samples, r)``.
    weights : torch.Tensor
        Statistical weights associated with each sample.
    r : int
        Number of learned representation components.
    friction : torch.Tensor
        Langevin friction-related prefactor used to weight position
        gradients.
    n_dim : int, default=3
        Number of spatial dimensions.
    descriptors_derivatives : SmartDerivatives or torch.Tensor, optional
        Descriptor derivatives with respect to atomic positions. Required
        when ``input`` contains descriptors instead of positions.
    ref_idx : torch.Tensor, optional
        Reference indices mapping the batch entries to the original dataset.
        Required when using ``SmartDerivatives``.

    Returns
    -------
    cov_X : torch.Tensor
        Weighted covariance matrix of the augmented representation, with
        shape ``(r + 1, r + 1)``.
    dcov_X : torch.Tensor
        Weighted gradient covariance matrix, with shape ``(r + 1, r + 1)``.

    Notes
    -----
    A constant column is appended internally to ``output`` before covariance
    computation.
    """
    if isinstance(input, torch_geometric.data.batch.Batch):
        _is_graph_data = True
        batch = torch.clone(input['batch'])
        node_types = torch.where(input['node_attrs'])[1]
        input = input['positions']
        device = input.device
    else:
        _is_graph_data=False
    device = input.device

    # check output and r
    if output.shape[-1] != r:
        raise ValueError ( 
            f"The number of eigenfunctions to compute (r) must match the number of outputs from the model! Found r:{r} and output.shape:{output.shape}"
            )
        
    one_column = torch.ones((output.shape[0],1),device=device)
    output = torch.cat((output,one_column),dim=1)
    # expand friction tensor
    if _is_graph_data:
        friction = friction[node_types].unsqueeze(1).unsqueeze(2)
    else:
        friction = friction = friction.unsqueeze(-1).repeat((1, n_dim)).ravel()
    # ------------------------ GRADIENTS ------------------------    
    # compute gradients of output wrt to the input iterating on the outputs
    grad_outputs = torch.ones(len(output), device=device)
    gradient = torch.stack([torch.autograd.grad(outputs=output[:, idx],
                                                inputs=input,
                                                grad_outputs=grad_outputs, 
                                                retain_graph=True, 
                                                create_graph=True)[0] for idx in range(r+1)
                            ], dim=2)
    
    # in case the input is not positions but descriptors, we need to correct the gradients up to the positions
    # --> If we pass a SmartDerivative object that takes the nonzero elements of the matrix d_desc/d_pos
    if isinstance(descriptors_derivatives, SmartDerivatives):
        gradient_positions = descriptors_derivatives(gradient, ref_idx).reshape(input.shape[0], -1, r+1)
    
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
    

    if r==1:
        gradient_positions = gradient_positions.unsqueeze(-1)

    # this is to make the following computation easier to write
    gradient_positions = gradient_positions.swapaxes(2,1)
    #if cell is not None:
    #    gradient_positions /= cell.repeat_interleave(gradient_positions.shape[-1]//n_dim)
    # multiply by friction

    try:
        gradient_positions = gradient_positions * torch.sqrt(friction)
    except RuntimeError as e:
        raise RuntimeError(e, """[HINT]: Is you system in 3 dimension? By default the code assumes so, if it's not the case change the n_dim key to the right dimensionality.""")

    if _is_graph_data:
        gradient_positions = torch.einsum("ikd,ild->ikl",gradient_positions, gradient_positions)
        gradient_positions = scatter_sum(gradient_positions, batch,dim=0)
        dcov_X = torch.einsum("ikl,i->kl",gradient_positions,weights)
    else:
        dcov_X = torch.einsum("ijk,ilk,i->jl", gradient_positions, gradient_positions, weights) 
    # ------------------------ COVARIANCES ------------------------
    # Compute covariances
    cov_X = torch.einsum("ik,il,i->kl",output,output,weights)
    del gradient_positions
    del gradient
    return cov_X.detach(), dcov_X.detach() 


def compute_eigenfunctions(dataset : DictDataset,
                           feature_method,
                           r : int,
                           eta : float,
                           friction : torch.Tensor,
                           tikhonov_reg : float = 1e-4,
                           descriptors_derivatives : Union[SmartDerivatives, torch.Tensor] = None,
                           n_dim : int = 3,
                           batch_size=None,
                           soft_max_postproc=True,
                           is_graph=False,
                           cell=None
                           ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute generator eigenfunctions from a learned representation.

    The function estimates the covariance and gradient covariance matrices of
    the learned representation, constructs the shifted generator-resolvent
    problem, and solves the resulting generalized eigenvalue problem.

    The returned eigenvalues correspond to the infinitesimal generator, not
    directly to the resolvent. If ``mu`` is a resolvent eigenvalue, the
    generator eigenvalue is computed as:

    .. math::

        \\lambda = \\eta \\left(1 - \\frac{1}{\\mu}\\right)

    Parameters
    ----------
    dataset : DictDataset
        Dataset containing ``"data"`` and ``"weights"``. For graph models,
        the dataset must provide graph inputs through ``get_graph_inputs``.
    feature_method : 
        Method to compute the features, usually the forward_nn of a DeepGenerator class
    r : int
        Number of learned representation components.
    eta : float
        Resolvent shift parameter.
    friction : torch.Tensor
        Langevin friction-related prefactor used to weight gradients.
    tikhonov_reg : float, default=1e-4
        Regularization parameter kept for API compatibility. Currently the
        implementation uses a pseudo-inverse of the matrix square root.
    descriptors_derivatives : SmartDerivatives or torch.Tensor, optional
        Descriptor derivatives with respect to atomic positions.
    n_dim : int, default=3
        Number of spatial dimensions.
    batch_size : int, optional
        Batch size used to estimate covariances. Defaults to the full dataset.
    soft_max_postproc : bool, default=True
        Whether to apply softmax to the model output before covariance
        estimation.
    is_graph : bool, default=False
        Whether the dataset contains graph samples.

    Returns
    -------
    eigenfunctions : torch.Tensor
        Eigenfunctions evaluated on the dataset.
    evals : torch.Tensor
        Generator eigenvalues sorted in descending order.
    evecs : torch.Tensor
        Eigenvectors mapping the augmented learned representation to
        eigenfunctions.
    output : torch.Tensor
        Augmented model output used to compute the eigenfunctions.

    Notes
    -----
    A constant basis function is appended internally to the learned
    representation. The eigenvectors therefore have dimension ``r + 1`` in
    the augmented basis.
    """

    # ------------------------ SETUP ------------------------
    # get device
    

    if batch_size is None:
        batch_size = dataset["weights"].shape[0]
    if is_graph:
        loader = torch_geometric.loader.DataLoader(dataset, 
                                                   batch_size=batch_size, 
                                                   shuffle=False )
        input = dataset.get_graph_inputs()
        weights = input['weight']
    else:
        loader = torch.utils.data.DataLoader(dataset, 
                                                   batch_size=batch_size, 
                                                   shuffle=False )
        weights = dataset["weights"]
    
    covariance = torch.zeros((r+1,r+1),device=weights.device)
    dcov = torch.zeros((r+1,r+1),device=weights.device)
    output = torch.zeros((len(weights),r+1),device=weights.device)

    for i,batch in enumerate(loader):
        print(f"Processing batch {i}/{len(loader)}", end='\r')
        batch_start, batch_stop = i*batch_size, (i+1) * batch_size
        if is_graph:
            batch_input = batch["data_list"]
            batch_input["positions"].requires_grad=True
            batch_input['node_attrs'].requires_grad=True
            batch_weights = batch_input["weight"]
            ref_idx = None
            cell = None
        else:
            batch_input = batch["data"]
            batch_weights = batch["weights"]
            batch_input.requires_grad = True
            if isinstance(descriptors_derivatives, SmartDerivatives):
                ref_idx = batch["ref_idx"]
            ### BATCHING derivatives is pretty handy, and faster than smart derivatives
            #elif "derivatives" in batch.keys():
            #    ref_idx = None
            #    descriptors_derivatives=batch["derivatives"]
            else:
                ref_idx=None
        if cell is not None:
            batch_output = feature_method(batch_input, cell=cell)
        else:
            batch_output = feature_method(batch_input) 
        if soft_max_postproc:
            batch_output = torch.nn.functional.softmax(batch_output,dim=-1)

        cov_batch, dcov_batch = compute_covariances(input=batch_input,
                                                    output=batch_output,
                                                    weights=batch_weights,
                                                    r=batch_output.shape[1],
                                                    friction=friction,
                                                    descriptors_derivatives=descriptors_derivatives,
                                                    ref_idx=ref_idx,
                                                    n_dim=n_dim

        )
        one_column = torch.ones((batch_output.shape[0],1),device=batch_output.device)
        batch_output = torch.cat((batch_output,one_column),dim=1)
        output[batch_start:batch_stop] = batch_output
        covariance += cov_batch.detach()
        dcov += dcov_batch.detach()
        del batch_input
        del cov_batch
        del dcov_batch
        del batch_output
        gc.collect()

    npts = len(weights)
    #x = input["positions"].reshape(weights.shape[0],input["positions"].shape[0]//weights.shape[0]*3)
    covariance /= len(weights)
    dcov /=  len(weights)
    W = covariance + dcov/eta + tikhonov_reg * torch.eye(covariance.shape[0], device = covariance.device)
    W_sq_inv = torch.linalg.pinv(sqrtmh(W))
    M = W_sq_inv @ covariance @ W_sq_inv
    evals, evecs = torch.linalg.eigh(M)
    evecs = W_sq_inv @ evecs
    
    idx = torch.argsort(-evals)
    evals = evals[idx]
    evecs = evecs[:, idx]

    numerically_nonzero_values_idxs = evals > torch.finfo(evals.dtype).eps
    evals = eta * (1 - 1 / evals[numerically_nonzero_values_idxs])
    evecs = evecs[:, numerically_nonzero_values_idxs]

    evecs_norm = torch.sqrt(((covariance @ evecs) * evecs).sum(0)) 
    stable_norms_idxs = evecs_norm > torch.finfo(evals.dtype).eps
    rank = np.min([stable_norms_idxs.shape[0],r+1])
    energy_scaling = evecs_norm[stable_norms_idxs][:rank]
    evecs = evecs[:, stable_norms_idxs][:, :rank]
    evals = evals[stable_norms_idxs][:rank]
    evecs /= energy_scaling
    eigenfunctions = (output @ evecs )

    return eigenfunctions, evals.detach(), evecs, output



# For the future, it might be worth having a more general function
def forecast_state_occupation(eigenfunctions: torch.Tensor,
                              eigenvalues: torch.Tensor,
                              times: torch.Tensor,
                              classification: torch.Tensor,
                              weights: torch.Tensor,
                              n_states: int,
                              reg_first_mode: bool = True,
                              ) -> torch.Tensor:
    """
    Computes the time evolution of state occupation probabilities in a dynamical system from the learned eigenfunctions.

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
        Biasing weights
    n_states : int
        The total number of discrete states in the system.
    reg_first_mode : bool
        Whether to regularize the first mode to have eigenfunction equal to 1 and eigenvalue equal to 0 to stabilize the calculation, by default True. 

    Returns:
    --------
    occupations : torch.Tensor, shape (n_states, n_states, n_times)
        A tensor where `occupations[i, j, t]` represents the probability of transitioning
        from state `i` to state `j` at time `times[t]`.

    """
    if reg_first_mode:
        eigenfunctions[:, 0] = 1
        eigenvalues[0] = 0

    # Number of samples
    n_samples = classification.shape[0]

    # Create masks for each state
    state_masks = torch.arange(n_states, device=classification.device).view(-1, 1) == classification.unsqueeze(0)  # (n_states, N)

    # Compute initial state occupations u
    inv_u_0 = (state_masks * weights).mean(dim=1, keepdim=True)  # (n_states, 1)
    u_0 = state_masks / inv_u_0

    # Project onto eigenfunctions
    initial_state_on_basis = ((u_0 * weights) @ eigenfunctions) / n_samples  # (n_states, n_eigen)
    final_state_on_basis = ((state_masks * weights) @ eigenfunctions) / n_samples  # Ensure proper mean normalization

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



# ---------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------- TESTS ----------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------

def test_forecast_state_occupation():
    # reference eigenfunctions
    eigenfunctions = torch.Tensor([[-1.0000, -0.1565, 1.0694],
                                   [-1.0000, -0.1567, 1.1080],
                                   [-1.0000, -0.1565, 1.1119],
                                   [-1.0000, -0.1567, -0.3890],
                                   [-0.9999, 6.3998, -0.0503]]
                                   )
    # reference eigenvalues
    evals = torch.Tensor([-4.9422e-05, -2.2918e-04, -1.1490e-01])

    # labels of points used, one for each eigenfunction
    classification = torch.Tensor([1, 1, 1, 0, 2])
    times = torch.linspace(0, 100, 10)
    weights = torch.Tensor([1.4809, 0.0736, 0.3693, 0.1849, 0.0885])
    ref_occupation_numbers = torch.Tensor([[[4.3484e-02, 3.9426e-02, 3.8278e-02, 3.7942e-02, 3.7832e-02, 
                                             3.7785e-02, 3.7755e-02, 3.7731e-02, 3.7708e-02, 3.7685e-02],
                                            [2.3270e-01, 3.4891e-01, 3.8116e-01, 3.8998e-01, 3.9228e-01, 
                                             3.9275e-01, 3.9271e-01, 3.9253e-01, 3.9231e-01, 3.9208e-01],
                                            [2.9414e-04, 7.9851e-05, 4.5504e-05, 6.1280e-05, 9.0959e-05, 
                                             1.2444e-04, 1.5890e-04, 1.9355e-04, 2.2819e-04, 2.6274e-04]],
                                           [[2.2365e-02, 3.3534e-02, 3.6634e-02, 3.7482e-02, 3.7703e-02, 
                                             3.7748e-02, 3.7744e-02, 3.7727e-02, 3.7706e-02, 3.7684e-02],
                                            [8.4217e-01, 5.1892e-01, 4.2858e-01, 4.0321e-01, 3.9596e-01,
                                             3.9377e-01, 3.9299e-01, 3.9260e-01, 3.9232e-01, 3.9208e-01],
                                            [-9.9108e-04, -2.6303e-04, -3.4530e-05, 5.4529e-05, 1.0461e-04,
                                             1.4374e-04, 1.7974e-04, 2.1479e-04, 2.4949e-04, 2.8402e-04]],
                                           [[6.1453e-04, 1.6683e-04, 9.5072e-05, 1.2803e-04, 1.9004e-04,
                                             2.5998e-04, 3.3197e-04, 4.0438e-04, 4.7674e-04, 5.4893e-04],
                                            [-2.1544e-02, -5.7176e-03, -7.5061e-04, 1.1853e-03, 2.2740e-03, 
                                             3.1246e-03, 3.9072e-03, 4.6690e-03, 5.4234e-03, 6.1740e-03],
                                            [7.4269e-01, 7.4080e-01, 7.3894e-01, 7.3710e-01, 7.3526e-01,
                                             7.3342e-01, 7.3159e-01, 7.2977e-01, 7.2795e-01, 7.2613e-01]]]
    )

    occupation_numbers = forecast_state_occupation(eigenfunctions, evals, times, classification, weights, 3)
    
    # check we are all good
    assert torch.allclose(occupation_numbers, ref_occupation_numbers, atol=1e-2)
