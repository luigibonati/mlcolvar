import torch
from warnings import warn

__all__ = ["TDA_loss"]

def TDA_loss(H : torch.tensor,
            labels : torch.tensor,
            n_states : int,
            target_centers : list or torch.tensor,
            target_sigmas : list or torch.tensor,
            alfa : float = 1,
            beta : float = 100) -> torch.tensor:
    """
    Compute a loss function as the distance from a simple Gaussian target distribution.
    
    Parameters
    ----------
    H : torch.tensor
        Output of the NN
    labels : torch.tensor
        Labels of the dataset
    n_states : int
        Number of states in the target
    target_centers : list or torch.tensor
        Centers of the Gaussian targets
        Shape: (n_states, n_cvs)
    target_sigmas : list or torch.tensor
        Standard deviations of the Gaussian targets
        Shape: (n_states, n_cvs)
    alfa : float, optional
        Centers_loss component prefactor, by default 1
    beta : float, optional
        Sigmas loss compontent prefactor, by default 100

    Returns
    -------
    torch.tensor
        Total loss, centers loss, sigmas loss
    """
    if type(target_centers) is list:
        target_centers = torch.tensor(target_centers)
    if type(target_sigmas) is list:
        target_sigmas = torch.tensor(target_sigmas)
    
    loss_centers = torch.zeros_like(target_centers, dtype = torch.float32)
    loss_sigmas = torch.zeros_like(target_sigmas, dtype = torch.float32)
    for i in range(n_states):
        # check which elements belong to class i
        if not torch.nonzero(labels == i).any():
            raise ValueError(f'State {i} was not represented in this batch! Either use bigger batch_size or a more equilibrated dataset composition!')
        else:
            H_red = H[torch.nonzero(labels == i).view(-1)]

            # compute mean and standard deviation over the class i
            mu = torch.mean(H_red, 0)
            if len(torch.nonzero(labels == i)) == 1:
                warn(f'There is only sample for state {i} in this batch! Std is set to 0, this may affect the training! Either use bigger batch_size or a more equilibrated dataset composition!')
                sigma = 0
            else:
                sigma = torch.std(H_red, 0)

        # compute loss function contributes for class i
        loss_centers[i] = alfa*(mu - target_centers[i]).pow(2)
        loss_sigmas[i] = beta*(sigma - target_sigmas[i]).pow(2)
        
        
    # get total model loss   
    loss_centers = torch.sum(loss_centers)
    loss_sigmas = torch.sum(loss_sigmas) 
    loss = loss_centers + loss_sigmas  

    return loss, loss_centers, loss_sigmas

