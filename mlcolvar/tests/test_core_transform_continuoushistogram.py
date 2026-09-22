import torch

from mlcolvar.core.transform.tools.continuous_hist import ContinuousHistogram


def test_continuous_histogram():
    x = torch.randn((5, 100))

    x.requires_grad = True

    hist = ContinuousHistogram(
        in_features=100,
        min=-1,
        max=1,
        bins=10,
        sigma_to_center=1,
    )

    out = hist(x)

    out.sum().backward()