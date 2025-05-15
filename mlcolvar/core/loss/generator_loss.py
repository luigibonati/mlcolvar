__all__ = [
    "GeneratorLoss",
    "compute_eigenfunctions",
    "compute_covariance",
    "forecast_state_occupation",
]

# =============================================================================
# GLOBAL IMPORTS
# =============================================================================

import torch


class GeneratorLoss(torch.nn.Module):
    """
    Computes the loss to learn a representation for the generator
    """

    def __init__(self, eta, cell, friction, alpha, n_cvs):
        """
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
        self.eta = eta
        self.friction = friction
        self.lambdas = torch.nn.Parameter(10 * torch.randn(n_cvs), requires_grad=True)
        self.alpha = alpha
        self.cell = cell

    def forward(self, data, output, weights, gradient_descriptors=None):
        return generator_loss(
            data,
            output,
            self.eta,
            self.alpha,
            self.friction,
            self.cell,
            self.lambdas,
            weights,
            gradient_descriptors,
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


def generator_loss(
    data,
    output,
    eta,
    alpha,
    friction,
    cell,
    lambdas,
    weights,
    gradient_descriptors=None,
):
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
    lambdas = lambdas**2
    diag_lamb = torch.diag(lambdas)
    # sorted_lambdas = lambdas[torch.argsort(lambdas)]
    r = output.shape[1]
    sample_size = output.shape[0] // 2

    gradient = torch.stack(
        [
            torch.autograd.grad(
                outputs=output[:, idx].sum(),
                inputs=data,
                retain_graph=True,
                create_graph=True,
            )[0]
            for idx in range(r)
        ],
        dim=2,
    ).swapaxes(2, 1)
    ### jacrev seems to be a bit faster, but I need to pass the model as argument, and in order to go with the philosophy of the other losses I am keeping it like this for now
    # compute_batch_jacobian = functorch.vmap(functorch.jacrev(model,argnums=0),in_dims=(0))
    # gradient = compute_batch_jacobian(data.unsqueeze(1))
    gradient = gradient.reshape(weights.shape[0], output.shape[1], -1)

    if gradient_descriptors is None:  # If there is no descriptors or a precompute layer
        gradient_positions = gradient * torch.sqrt(friction)
    else:
        gradient_positions = torch.einsum(
            "ijk,imkl->ijml", gradient, gradient_descriptors
        )
        gradient_positions = gradient_positions.reshape(
            -1, output.shape[1], gradient_descriptors.shape[1] * 3
        ) * torch.sqrt(friction)
    if cell is not None:
        gradient_positions /= cell

    ### In order to have unbiased estimation, we split the dataset in two chunks ###
    weights_X, weights_Y = weights[:sample_size], weights[sample_size:]
    gradient_X, gradient_Y = (
        gradient_positions[:sample_size],
        gradient_positions[sample_size:],
    )
    psi_X, psi_Y = output[:sample_size], output[sample_size:]

    cov_X = compute_covariance(psi_X, weights_X)

    cov_Y = compute_covariance(psi_Y, weights_Y)

    dcov_X = compute_covariance(gradient_X, weights_X)

    dcov_Y = compute_covariance(gradient_Y, weights_Y)

    W1 = (eta * cov_X + dcov_X) @ diag_lamb
    W2 = (eta * cov_Y + dcov_Y) @ diag_lamb

    ### Unbiased estimation of the "variational part"
    # It might be worse replacing with einsum if it is faster
    loss_ef = torch.trace(
        ((cov_X @ diag_lamb) @ W2 + (cov_Y @ diag_lamb) @ W1) / 2
        - cov_X @ diag_lamb
        - cov_Y @ diag_lamb
    )

    # Compute loss_ortho
    loss_ortho = alpha * (
        torch.trace(
            (torch.eye(output.shape[1], device=output.device) - cov_X).T
            @ (torch.eye(output.shape[1], device=output.device) - cov_Y)
        )
    )

    loss = loss_ef + loss_ortho  # loss_ortho
    return loss, loss_ef, loss_ortho


def compute_eigenfunctions(
    input,
    output,
    weights,
    friction,
    eta,
    r,
    cell=None,
    tikhonov_reg=1e-4,
    descriptors_derivatives=None,
):
    """
    Computes eigenfunctions and eigenvalues from a learned representation.

    This function estimates the eigenfunctions and eigenvalues of the infinitesimal generator
    associated with the Langevin process. The eigenvalues are computed using a resolvent approach,
    where `evals` relate to the generator's eigenvalues as: `lambda = eta - 1/evals`.

    Parameters:
    ----------
    input : torch.Tensor, shape (N,d)
        Input tensor containing the data
    output : torch.Tensor, shape (N,r)
        Output tensor containing the learned representation of the data
    weights : torch.Tensor, shape (N,)
        Biasing weights (set to one for unbiased simulations)
    friction : torch.Tensor, shape (N,)
        Langevin friction values for each data point.
    eta : float
        Hyperparameter for the resolvent approach.
    r : int
        Number of eigenfunctions to compute.

    cell : torch.Tensor, optional, shape (3,3)
        If provided, used to normalize the gradients when periodic boundary conditions apply.
    descriptors_derivatives : torch.Tensor, optional, shape (N, natoms, d, 3)
        Derivatives of descriptors with respect to atomic positions. If `None`,
        the function uses direct gradients of `model(X)`.
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
    # friction=friction.to("cuda")

    d = input.shape[1]

    # gradient with respect to descriptors
    gradient_X = torch.stack(
        [
            torch.autograd.grad(
                outputs=output[:, idx].sum(),
                inputs=input,
                retain_graph=True,
                create_graph=True,
            )[0].reshape((-1, d))
            for idx in range(r)
        ],
        dim=2,
    ).swapaxes(2, 1)

    if descriptors_derivatives is not None:  # If there is no precomputing layer
        gradient_positions = torch.einsum(
            "ijk,imkl->ijml", gradient_X, descriptors_derivatives
        )
        gradient_positions = gradient_positions.reshape(
            -1, output.shape[1], descriptors_derivatives.shape[1] * 3
        ) * torch.sqrt(friction)

    else:
        gradient_positions = gradient_X * torch.sqrt(friction)
    if cell is not None:
        gradient_positions /= cell

    weights_X = weights

    # Compute covariances
    cov_X = compute_covariance(output, weights_X)
    dcov_X = compute_covariance(gradient_positions, weights_X)

    W = eta * cov_X + dcov_X
    # The resolvent projected on the learned space
    operator = (
        torch.linalg.inv(
            W + tikhonov_reg * torch.eye(output.size(1), device=output.device)
        )
        @ cov_X
    )
    evals, evecs = torch.linalg.eig(operator)
    # eigenfunctions
    g = output @ evecs.real
    lambdas = eta - 1 / evals  # eigenvalues of the generator
    sorting = torch.argsort(-lambdas.real)
    # Ensure normalization of eigenfunctions
    detached_evecs = evecs.detach()
    detached_evecs /= torch.sqrt(torch.mean(weights.unsqueeze(1) * g**2, axis=0))
    g /= torch.sqrt(torch.mean(weights.unsqueeze(1) * g**2, axis=0))
    return g[:, sorting], lambdas.detach()[sorting], detached_evecs.detach()[:, sorting]


# For the future, it might be worse having a more general function
def forecast_state_occupation(
    eigenfunctions: torch.Tensor,
    eigenvalues: torch.Tensor,
    times: torch.Tensor,
    classification: torch.Tensor,
    weights: torch.Tensor,
    n_states: torch.Tensor,
):
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
    N = classification.shape[0]

    # Create masks for each state
    state_masks = torch.arange(n_states, device=classification.device).view(
        -1, 1
    ) == classification.unsqueeze(
        0
    )  # (n_states, N)

    # Compute initial state occupations
    inv_u_0 = (state_masks * weights).mean(dim=1, keepdim=True)  # (n_states, 1)
    u_0 = state_masks / inv_u_0
    # Project onto eigenfunctions
    initial_state_on_basis = (
        (u_0 * weights) @ eigenfunctions
    ) / N  # (n_states, n_eigen)
    final_state_on_basis = (
        (state_masks * weights) @ eigenfunctions
    ) / N  # Ensure proper mean normalization

    # Ensure eigenvalues are correctly shaped
    eigenvalues = eigenvalues.view(1, -1)  # (1, n_eigen)

    # Compute time evolution
    time_evolution = torch.exp(times.view(-1, 1) * eigenvalues)  # (n_times, n_eigen)

    # Compute occupation over time
    occupation_over_time = (
        (
            initial_state_on_basis[:, None, :] * final_state_on_basis[None, :, :]
        )  # (n_states, n_states, n_eigen)
        @ time_evolution.T.real  # Matrix multiplication over n_eigen -> (n_states, n_states, n_times)
    )

    return occupation_over_time  # Shape: (n_states, n_states, n_times)


def test_forecast_state_occupation():
    eigenfunctions = torch.Tensor(
        [
            [-1.0000, -0.1565, 1.0694],
            [-1.0000, -0.1567, 1.1080],
            [-1.0000, -0.1565, 1.1119],
            [-1.0000, -0.1567, -0.3890],
            [-0.9999, 6.3998, -0.0503],
        ]
    )
    evals = torch.Tensor([-4.9422e-05, -2.2918e-04, -1.1490e-01])

    classification = torch.Tensor([1, 1, 1, 0, 2])
    times = torch.linspace(0, 100, 10)
    weights = torch.Tensor([1.4809, 0.0736, 0.3693, 0.1849, 0.0885])
    ref_occupation_numbers = torch.Tensor(
        [
            [
                [
                    4.3484e-02,
                    3.9426e-02,
                    3.8278e-02,
                    3.7942e-02,
                    3.7832e-02,
                    3.7785e-02,
                    3.7755e-02,
                    3.7731e-02,
                    3.7708e-02,
                    3.7685e-02,
                ],
                [
                    2.3270e-01,
                    3.4891e-01,
                    3.8116e-01,
                    3.8998e-01,
                    3.9228e-01,
                    3.9275e-01,
                    3.9271e-01,
                    3.9253e-01,
                    3.9231e-01,
                    3.9208e-01,
                ],
                [
                    2.9414e-04,
                    7.9851e-05,
                    4.5504e-05,
                    6.1280e-05,
                    9.0959e-05,
                    1.2444e-04,
                    1.5890e-04,
                    1.9355e-04,
                    2.2819e-04,
                    2.6274e-04,
                ],
            ],
            [
                [
                    2.2365e-02,
                    3.3534e-02,
                    3.6634e-02,
                    3.7482e-02,
                    3.7703e-02,
                    3.7748e-02,
                    3.7744e-02,
                    3.7727e-02,
                    3.7706e-02,
                    3.7684e-02,
                ],
                [
                    8.4217e-01,
                    5.1892e-01,
                    4.2858e-01,
                    4.0321e-01,
                    3.9596e-01,
                    3.9377e-01,
                    3.9299e-01,
                    3.9260e-01,
                    3.9232e-01,
                    3.9208e-01,
                ],
                [
                    -9.9108e-04,
                    -2.6303e-04,
                    -3.4530e-05,
                    5.4529e-05,
                    1.0461e-04,
                    1.4374e-04,
                    1.7974e-04,
                    2.1479e-04,
                    2.4949e-04,
                    2.8402e-04,
                ],
            ],
            [
                [
                    6.1453e-04,
                    1.6683e-04,
                    9.5072e-05,
                    1.2803e-04,
                    1.9004e-04,
                    2.5998e-04,
                    3.3197e-04,
                    4.0438e-04,
                    4.7674e-04,
                    5.4893e-04,
                ],
                [
                    -2.1544e-02,
                    -5.7176e-03,
                    -7.5061e-04,
                    1.1853e-03,
                    2.2740e-03,
                    3.1246e-03,
                    3.9072e-03,
                    4.6690e-03,
                    5.4234e-03,
                    6.1740e-03,
                ],
                [
                    7.4269e-01,
                    7.4080e-01,
                    7.3894e-01,
                    7.3710e-01,
                    7.3526e-01,
                    7.3342e-01,
                    7.3159e-01,
                    7.2977e-01,
                    7.2795e-01,
                    7.2613e-01,
                ],
            ],
        ]
    )

    occupation_numbers = forecast_state_occupation(
        eigenfunctions, evals, times, classification, weights, 3
    )
    print(occupation_numbers)
    assert torch.allclose(occupation_numbers, ref_occupation_numbers, atol=1e-3)
