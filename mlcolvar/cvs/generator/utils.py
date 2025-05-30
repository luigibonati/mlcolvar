import torch
from typing import Union, Tuple
from mlcolvar.core.loss.generator_loss import compute_covariance
from mlcolvar.core.loss.committor_loss import SmartDerivatives

__all__ = ["compute_eigenfunctions", "forecast_state_occupation"]

def compute_eigenfunctions(input : torch.Tensor,
                           output : torch.Tensor,
                           weights : torch.Tensor,
                           r : int, # TODO add check on dimensions
                           eta : float,
                           friction : torch.Tensor,
                           cell: float = None,
                           tikhonov_reg : float = 1e-4,
                           descriptors_derivatives : Union[SmartDerivatives, torch.Tensor] = None
                           ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Computes eigenfunctions and eigenvalues from a learned representation.

    This function estimates the eigenfunctions and eigenvalues of the infinitesimal generator
    associated with the Langevin process. The eigenvalues are computed using a resolvent approach,
    where `evals` relate to the generator's eigenvalues as: `lambda = eta - 1/evals`.

    Parameters
    ----------
    input : torch.Tensor
        Input of the model
    output : torch.Tensor
        Output containing the learned representation of the data, i.e., output of the (set of) neural networks
    weights : torch.Tensor
        Statistical weights of the samples, this could be from reweighting.
    r : int
        Number of eigenfunctions to compute.
    eta : float
        Hyperparameter for the shift to define the resolvent, i.e., $(\eta I-_mathcal{L})^{-1}$
    friction : torch.Tensor
        Langevin friction, i.e., $\sqrt{k_B*T/(gamma*m_i)}$
    cell : float, optional
        CUBIC cell size length, used to scale the positions from reduce coordinates to real coordinates, by default None
    tikhonov_reg : float, optional
        Hyperparameter for the regularization of the inverse (Ridge regression parameter), by default 1e-4
    descriptors_derivatives : Union[SmartDerivatives, torch.Tensor], optional
        Derivatives of descriptors wrt atomic positions (if used) to speed up calculation of gradients, by default None. 
        Can be either:
            - A `SmartDerivatives` object to save both memory and time, see also mlcolvar.core.loss.committor_loss.SmartDerivatives
            - A torch.Tensor with the derivatives to save time, memory-wise could be less efficient

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        g : torch.Tensor, shape (N, r)
            The computed eigenfunctions evaluated at each data point.
        lambdas : torch.Tensor, shape (r,)
            The eigenvalues associated with the generator, sorted in descending order.
        evecs : torch.Tensor, shape (r, r)
            The eigenvectors of the operator.
    
    Notes:
    ------
    - Eigenfunctions are normalized using the dataset weights.
    - The operator matrix is regularized to improve numerical stability.
    """

    # ------------------------ SETUP ------------------------
    # get device
    device = input.device

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
    # Compute covariances
    cov_X = compute_covariance(output, weights)
    dcov_X = compute_covariance(gradient_positions, weights)

    # TODO what is this?
    W = eta * cov_X + dcov_X

    # The resolvent projected on the learned space
    operator = (
        torch.linalg.inv(
            W + tikhonov_reg * torch.eye(output.size(1), device=output.device)
        )
        @ cov_X
    )


    # ------------------------ EIGENFUNCTIONS ------------------------

    # get eigenvalues and eigenvectors of resolvent # TODO correct?
    evals, evecs = torch.linalg.eigh(operator)

    # eigenfunctions and eigenvalues of generator
    lambdas = eta - 1 / evals
    sorting = torch.argsort(-lambdas)
    
    # eigenfunctions of generator
    eigenfunctions = output @ evecs
    
    # Ensure normalization of eigenfunctions
    detached_evecs = evecs.detach()
    detached_evecs /= torch.sqrt( torch.mean( weights.unsqueeze(1) * eigenfunctions**2, axis=0 ) )
    eigenfunctions /= torch.sqrt( torch.mean( weights.unsqueeze(1) * eigenfunctions**2, axis=0 ) )

    return eigenfunctions[:, sorting], lambdas.detach()[sorting], detached_evecs.detach()[:, sorting]


# For the future, it might be worse having a more general function
def forecast_state_occupation(eigenfunctions: torch.Tensor,
                              eigenvalues: torch.Tensor,
                              times: torch.Tensor,
                              classification: torch.Tensor,
                              weights: torch.Tensor,
                              n_states: int
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

    Returns:
    --------
    occupations : torch.Tensor, shape (n_states, n_states, n_times)
        A tensor where `occupations[i, j, t]` represents the probability of transitioning
        from state `i` to state `j` at time `times[t]`.

    """

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

    # TODO what?
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

if __name__ == '__main__':
    test_forecast_state_occupation()