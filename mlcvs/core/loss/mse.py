import torch

__all__ = ['MSE_loss']

def MSE_loss(diff : torch.Tensor, options = {}):
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
    if 'weights' in options:
        w = options['weights']
        if w.ndim == 1:
            w = w.unsqueeze(1)
        loss = (diff*w).square().mean()
    else:
        loss = diff.square().mean()
    return loss