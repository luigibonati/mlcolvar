import torch
import torch_geometric
from typing import Union, Tuple
from mlcolvar.core.loss.utils.smart_derivatives import SmartDerivatives
from mlcolvar.data import DictDataset, DictLoader
from mlcolvar.utils._code import scatter_sum
import numpy as np

import gc


def sqrtmh(A: torch.Tensor):
    L, Q = torch.linalg.eigh(A)
    zero = torch.zeros((), device=L.device, dtype=L.dtype)
    threshold = L.max(-1).values * L.size(-1) * torch.finfo(L.dtype).eps
    L = L.where(L > threshold.unsqueeze(-1), zero)  # zero out small components
    return (Q * L.sqrt().unsqueeze(-2)) @ Q.mH

def compute_covariances(input,output,weights,r,friction,n_dim=3,descriptors_derivatives : Union[SmartDerivatives, torch.Tensor] = None, ref_idx=None, softmax_postproc=True):
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
    if softmax_postproc:
        one_column = torch.ones((output.shape[0],1),device=device)
        output = torch.cat((output,one_column),dim=1)
        r = r + 1
    # expand friction tensor
    if _is_graph_data:
        friction = friction[node_types].unsqueeze(1).unsqueeze(2)
    else:
        friction = friction.unsqueeze(-1).repeat((1, n_dim)).ravel()
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
        gradient_positions = descriptors_derivatives(gradient, ref_idx).reshape(input.shape[0], -1, r)
    
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


def compute_eigenfunctions(dataloader : Union[DictLoader, torch_geometric.loader.DataLoader],
                           feature_method,
                           r : int,
                           eta : float,
                           friction : torch.Tensor,
                           tikhonov_reg : float = 1e-4,
                           descriptors_derivatives : Union[SmartDerivatives, torch.Tensor] = None,
                           n_dim : int = 3,
                           softmax_postproc=True,
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
    

    if softmax_postproc:
        r = r + 1
    n_points = len(dataloader.dataset)
    batch_size = dataloader.batch_size
    if batch_size==0:
        batch_size = n_points
    device = friction.device
    covariance = torch.zeros((r,r),device=device)
    dcov = torch.zeros((r,r),device=device)
    output = torch.zeros((n_points,r),device=device)

    for i,batch in enumerate(dataloader):
        print(f"Processing batch {i}/{len(dataloader)}", end='\r')
        batch_start, batch_stop = i*batch_size, (i+1) * batch_size
        if is_graph:
            batch_input = batch["data_list"].to(device)
            batch_input["positions"].requires_grad=True
            batch_input['node_attrs'].requires_grad=True
            batch_weights = batch_input["weight"].to(device)
            ref_idx = None
            cell = None
        else:
            batch_input = batch["data"].to(device)
            batch_weights = batch["weights"].to(device)
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
        if softmax_postproc:
            batch_output = torch.nn.functional.softmax(batch_output, dim=-1)

        cov_batch, dcov_batch = compute_covariances(input=batch_input,
                                                    output=batch_output,
                                                    weights=batch_weights,
                                                    r=batch_output.shape[1],
                                                    friction=friction,
                                                    descriptors_derivatives=descriptors_derivatives,
                                                    ref_idx=ref_idx,
                                                    n_dim=n_dim,
                                                    softmax_postproc=softmax_postproc

        )
        if softmax_postproc:
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

    
    #x = input["positions"].reshape(weights.shape[0],input["positions"].shape[0]//weights.shape[0]*3)
    covariance /= n_points
    dcov /=  n_points
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
def forecast_observable_evolution(eigenfunctions: torch.Tensor,
                              eigenvalues: torch.Tensor,
                              times: torch.Tensor,
                              observable: torch.Tensor,
                              initial_state: torch.Tensor,
                              weights: torch.Tensor,
                              reg_first_mode: bool = True,
                              ) -> torch.Tensor:
    """
    Computes the time evolution of an observable in a dynamical system from the learned eigenfunctions.

    

    Parameters:
    -----------
    eigenfunctions : torch.Tensor, shape (N, r)
        The eigenfunctions evaluated at each sample point, where N is the number of samples
        and r is the number of eigenfunctions.
    eigenvalues : torch.Tensor, shape (r,)
        The eigenvalues associated with the eigenfunctions.
    times : torch.Tensor, shape (n_times,)
        A 1D tensor containing the time points at which to evaluate the occupation probabilities.
    observable : torch.Tensor, shape (N,d)
        The observable you want to compute the time evolution of
    initial_state : torch.Tensor, shape (N,)
        A tensor assigning each sample point a value for the initial state (0 this frame does not correspond to the initial state, 1 this frame does).
    weights : torch.Tensor, shape (N,)
        Biasing weights
.
    reg_first_mode : bool
        Whether to regularize the first mode to have eigenfunction equal to 1 and eigenvalue equal to 0 to stabilize the calculation, by default True. 

    Returns:
    --------
    observable_time : torch.Tensor, shape (n_states, n_states, n_times)
        A tensor where `observable_time[t]` represents the evolution of the observable
         at time `times[t]`.

    """
    if reg_first_mode:
        eigenfunctions[:, 0] = 1
        eigenvalues[0] = 0

    # Number of samples
    n_samples = observable.shape[0]

    # Create masks for each state

    # Compute initial state occupations u
    inv_u_0 = (initial_state * weights).mean() # (n_states, 1)
    u_0 = initial_state / inv_u_0

    # Project onto eigenfunctions
    initial_state_on_basis = torch.einsum("i,i,ij->j",u_0,weights, eigenfunctions) / n_samples
    observable_on_basis = torch.einsum("id, i, ij->jd", observable, weights, eigenfunctions) / n_samples

    # Compute time evolution
    time_evolution = torch.exp(times[:,None] * eigenvalues) 
    
    observable_time = torch.einsum("i, id,ti->td", initial_state_on_basis, observable_on_basis, time_evolution.real)  

    return observable_time  # Shape: (n_states, n_states, n_times)

