import torch

__all__ = ["TDA_loss"]

def TDA_loss(H : torch.tensor,
            labels : torch.tensor,
            n_states : int,
            target_centers : list,
            target_sigmas : list,
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
    target_centers : list
        Centers of the Gaussian targets
        Shape: (n_states, n_cvs)
    target_sigmas : list
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
    
    target_centers = torch.tensor(target_centers)
    target_sigmas = torch.tensor(target_sigmas)
    loss_centers = torch.zeros_like(target_centers, dtype = torch.float32)
    loss_sigmas = torch.zeros_like(target_sigmas, dtype = torch.float32)
    
    for i in range(n_states):
        # check which elements belong to class i
        H_red = H[torch.nonzero(labels == i).view(-1)]

        # compute mean over the class i
        mu = torch.mean(H_red, 0)
        # compute standard deviation over class i
        sigma = torch.std(H_red, 0)

        # compute loss function contribute for class i
        loss_centers[i] = alfa*(mu - target_centers[i]).pow(2)
        loss_sigmas[i] = beta*(sigma - target_sigmas[i]).pow(2)
       
    loss_centers = torch.sum(loss_centers)
    loss_sigmas = torch.sum(loss_sigmas) 
    loss = loss_centers + loss_sigmas  

    return loss, loss_centers, loss_sigmas

