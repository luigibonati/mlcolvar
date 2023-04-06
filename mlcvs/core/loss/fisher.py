import torch
from mlcvs.core.stats import LDA

__all__ = ['fisher_discriminant_loss']

def fisher_discriminant_loss(X : torch.Tensor, labels : torch.Tensor, invert_sign = True):
    """ Fisher's discriminant ratio.

    $$L = - \frac{S_b(X)}{S_w(X)}$$
    
    Parameters
    ----------
    X : torch.Tensor
        input variable
    labels : torch.Tensor
        classes labels
    invert_sign: bool, optional
        whether to return the opposite of the function (in order to be minimized with GD methods), by default true

    Returns
    -------
    loss: torch.Tensor
        loss function
    """

    if X.ndim == 1:
        X = X.unsqueeze(1) # for lda compute_scatter_matrices method
    if X.ndim == 2 and X.shape[1] > 1:
        raise ValueError (f'fisher_discriminant_loss should be used on (batches of) scalar outputs, found tensor of shape {X.shape} instead.')

    # get params
    d = X.shape[-1] if X.ndim == 2 else 1
    n_classes = len(labels.unique())

    # define LDA object to compute S_b / S_w ratio
    lda = LDA(in_features=d,n_states=n_classes)

    s_b, s_w = lda.compute_scatter_matrices(X,labels)

    loss = s_b.squeeze() / s_w.squeeze()

    if invert_sign:
        loss *= -1

    return loss