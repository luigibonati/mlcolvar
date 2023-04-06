#!/usr/bin/env python

# =============================================================================
# MODULE DOCSTRING
# =============================================================================

"""
Reduce eigenvalues loss.
"""

__all__ = ['ReduceEigenvaluesLoss', 'reduce_eigenvalues_loss']


# =============================================================================
# GLOBAL IMPORTS
# =============================================================================

import torch


# =============================================================================
# LOSS FUNCTIONS
# =============================================================================

class ReduceEigenvaluesLoss(torch.nn.Module):
    """Calculate a monotonic function f(x) of the eigenvalues, by default the sum.

    By default it returns -f(x) to be used as loss function to maximize
    eigenvalues in gradient descent schemes.

    The following reduce functions are implemented:
        - sum     : sum_i (lambda_i)
        - sum2    : sum_i (lambda_i)**2
        - gap     : (lambda_1-lambda_2)
        - its     : sum_i (1/log(lambda_i))
        - single  : (lambda_i)
        - single2 : (lambda_i)**2

    """

    def __init__(
            self,
            mode: str = 'sum',
            n_eig: int = 0,
            invert_sign: bool = True,
    ):
        """Constructor.

        Parameters
        ----------
        mode : str, optional
            Function of the eigenvalues to optimize (see notes). Default is ``'sum'``.
        n_eig: int, optional
            Number of eigenvalues to include in the loss (default: 0 --> all).
            In case of ``'single'`` and ``'single2'`` is used to specify which
            eigenvalue to use.
        invert_sign: bool, optional
            Whether to return the opposite of the function (in order to be minimized
            with GD methods). Default is ``True``.
        """
        super().__init__()
        self.mode = mode
        self.n_eig = n_eig
        self.invert_sign = invert_sign

    def forward(self, evals: torch.Tensor) -> torch.Tensor:
        """Compute the loss.

        Parameters
        ----------
        evals : torch.Tensor
            Shape ``(n_batches, n_eigenvalues)``. Eigenvalues.

        Returns
        -------
        loss : torch.Tensor
        """
        return reduce_eigenvalues_loss(evals, self.mode, self.n_eig, self.invert_sign)


def reduce_eigenvalues_loss(
        evals: torch.Tensor,
        mode: str = 'sum',
        n_eig: int = 0,
        invert_sign: bool = True,
) -> torch.Tensor:
    """Calculate a monotonic function f(x) of the eigenvalues, by default the sum.

    By default it returns -f(x) to be used as loss function to maximize
    eigenvalues in gradient descent schemes.

    Parameters
    ----------
    evals : torch.Tensor
        Shape ``(n_batches, n_eigenvalues)``. Eigenvalues.
    mode : str, optional
        Function of the eigenvalues to optimize (see notes). Default is ``'sum'``.
    n_eig: int, optional
        Number of eigenvalues to include in the loss (default: 0 --> all).
        In case of ``'single'`` and ``'single2'`` is used to specify which
        eigenvalue to use.
    invert_sign: bool, optional
        Whether to return the opposite of the function (in order to be minimized
        with GD methods). Default is ``True``.

    Notes
    -----
    The following functions are implemented:
        - sum     : sum_i (lambda_i)
        - sum2    : sum_i (lambda_i)**2
        - gap     : (lambda_1-lambda_2)
        - its     : sum_i (1/log(lambda_i))
        - single  : (lambda_i)
        - single2 : (lambda_i)**2

    Returns
    -------
    loss : torch.Tensor (scalar)
        Loss value.
    """

    #check if n_eig is given and
    if (n_eig>0) & (len(evals) < n_eig):
        raise ValueError("n_eig must be lower than the number of eigenvalues.")
    elif (n_eig==0):
        if ( (mode == 'single') | (mode == 'single2')):
            raise ValueError("n_eig must be specified when using single or single2.")
        else:
            n_eig = len(evals)

    loss = None

    if   mode == 'sum':
        loss =  torch.sum(evals[:n_eig])
    elif mode == 'sum2':
        g_lambda =  torch.pow(evals,2)
        loss = torch.sum(g_lambda[:n_eig])
    elif mode == 'gap':
        loss =  (evals[0] -evals[1])
    elif mode == 'its':
        g_lambda = 1 / torch.log(evals)
        loss = torch.sum(g_lambda[:n_eig])
    elif mode == 'single':
        loss =  evals[n_eig-1]
    elif mode == 'single2':
        loss = torch.pow(evals[n_eig-1],2)
    else:
        raise ValueError(f"unknown mode : {mode}. options: 'sum','sum2','gap','single','its'.")

    if invert_sign:
        loss *= -1

    return loss
