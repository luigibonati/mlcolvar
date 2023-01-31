import torch

def batch_reshape(t: torch.Tensor, size : torch.Size) -> (torch.Tensor):
    """Return value reshaped according to size. 
    In case of batch expand unsqueeze and expand along the first dimension.
    For single inputs just pass:

    Parameters
    ----------
        Mean and range 

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
