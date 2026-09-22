import torch

from mlcolvar.core.loss.tda_loss import TDALoss


def test_tda_loss():
    H = torch.randn(100)
    H.requires_grad = True
    labels = torch.zeros_like(H)
    labels[-50:] = 1

    Loss = TDALoss(
        n_states=2,
        target_centers=[-1, 1],
        target_sigmas=[0.1, 0.1],
    )

    loss = Loss(H=H, labels=labels, return_loss_terms=True)

    loss[0].backward()