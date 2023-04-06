import torch

__all__ = ['autocorrelation_loss']

def autocorrelation_loss(z_t : torch.Tensor, z_lag : torch.Tensor, weights = None, invert_sign = True ):
    """(Weighted) autocorrelation loss.

    $$L = - \frac{\langle (x(t)-\bar{x}(t))(x(t+\tau)-\bar{x}(t)) \rangle}{\sigma(x_t)^2}$$
    
    Parameters
    ----------
    z_t : torch.Tensor
        values at time t
    z_lag : torch.Tensor
        values at time t+lag
    weights : torch.Tensor, optional
        sample weights, by default None
    invert_sign: bool, optional
        whether to return the opposite of the function (in order to be minimized with GD methods), by default true

    Returns
    -------
    loss: torch.Tensor
        loss function
    """

    if z_t.ndim == 2:
        if z_t.shape[1] > 1:
            raise ValueError (f'autocorrelation_loss should be used on (batches of) scalar outputs, found tensor of shape {z_t.shape} instead.')
        else:
            z_t = z_t.squeeze()
            z_lag = z_lag.squeeze()

    if weights is None:
        mean = z_t.mean()
        std = z_t.std()
        
        loss = ((z_t-mean)*(z_lag-mean)).mean()/std**2
    else:
        weights = weights.squeeze()
        weighted_mean = lambda x,w : (x*w).sum()/w.sum()

        mean = weighted_mean(z_t,weights)
        std = weighted_mean((z_t-mean)**2,weights).sqrt()
        
        loss = weighted_mean((z_t-mean)*(z_lag-mean), weights)/std**2
    
    if invert_sign:
        loss *= -1

    return loss