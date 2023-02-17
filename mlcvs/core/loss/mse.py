import torch

__all__ = ['MSE_loss']

def MSE_loss(diff : torch.Tensor, weights = None):
    """(Weighted) Mean Square Error

    Parameters
    ----------
    diff : torch.Tensor
        input-target
    options : dict, optional
        available options: 'weights', by default {}

    Returns
    -------
    loss: torch.Tensor
        loss function
    """
    if weights is not None:
        if weights.ndim == 1:
            weights = weights.unsqueeze(1)
        loss = (diff*weights).square().mean()
    else:
        loss = diff.square().mean()
    return loss