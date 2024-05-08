import torch
import numpy as np

from typing import Union, List

def batch_reshape(t: torch.Tensor, size: torch.Size) -> torch.Tensor:
    """Return value reshaped according to size.
    In case of batch unsqueeze and expand along the first dimension.
    For single inputs just pass.

    Parameters
    ----------
        mean and range

    """
    if len(size) == 1:
        return t
    if len(size) == 2:
        batch_size = size[0]
        x_size = size[1]
        t = t.unsqueeze(0).expand(batch_size, x_size)
    else:
        raise ValueError(
            f"Input tensor must of shape (n_features) or (n_batch,n_features), not {size} (len={len(size)})."
        )
    return t


def _gaussian_expansion(x : torch.Tensor, 
                        centers : torch.Tensor, 
                        sigma : torch.Tensor):
    """Computes the values in x of a set of Gaussian kernels centered on centers and with width sigma

    Parameters
    ----------
    x : torch.Tensor
        Input value(s)
    centers : torch.Tensor
        Centers of the Gaussian kernels
    sigma : torch.Tensor
        Width of the Gaussian kernels
    """
    return torch.exp(- torch.div(torch.pow(x-centers, 2), 2*torch.pow(sigma,2) ))

def easy_KDE(x : torch.Tensor, 
             n_input : int, 
             min_max : Union[List[float], np.ndarray], 
             n : int, 
             sigma_to_center : float = 1.0, 
             normalize : bool = False, 
             return_bins : bool = False) -> torch.Tensor:
    """Compute histogram using KDE with Gaussian kernels

    Parameters
    ----------
    x : torch.Tensor
        Input
    n_input : int
        Number of inputs per batch
    min_max : Union[list[float], np.ndarray]
        Minimum and maximum values for the histogram
    n : int
        Number of Gaussian kernels
    sigma_to_center : float, optional
        Sigma value in bin_size units, by default 1.0
    normalize : bool, optional
        Switch for normalization of the histogram to sum to n_input, by default False
    return_bins : bool, optional
        Switch to return the bins of the histogram alongside the values, by default False

    Returns
    -------
    torch.Tensor
        Values of the histogram for each bin. The bins can be optionally returned enabling `return_bins`.
    """
    if len(x.shape) == 1:
        x = torch.reshape(x, (1, n_input, 1))
    if x.shape[-1] != 1:
        x = x.unsqueeze(-1)
    if x.shape[0] == n_input:
        x = x.unsqueeze(0)

    centers = torch.linspace(min_max[0], min_max[1], n, device=x.device)
    bins = torch.clone(centers)
    sigma = (centers[1] - centers[0]) * sigma_to_center
    centers = torch.tile(centers, dims=(n_input,1))
    out = torch.sum(_gaussian_expansion(x, centers, sigma), dim=1)
    if normalize:
        out = torch.div(out, torch.sum(out, -1, keepdim=True)) * n_input
    if return_bins:
        return out, bins
    else:
        return out                   