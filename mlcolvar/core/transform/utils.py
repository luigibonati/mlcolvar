import torch
from typing import Union
from warnings import warn

__all__ = ["Inverse"]


class Inverse(torch.nn.Module):
    "Wrapper to return the inverse method of a module as a torch.nn.Module"

    def __init__(self, module: torch.nn.Module):
        """Return the inverse method of a module as a torch.nn.Module

        Parameters
        ----------
        module : torch.nn.Module
            Module to be inverted
        """
        super().__init__()
        if not hasattr(module, "inverse"):
            raise AttributeError("The given module does not have a 'inverse' method!")
        self.module = module

    def inverse(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def forward(self, *args, **kwargs):
        return self.module.inverse(*args, **kwargs)

def test_inverse():
    from mlcolvar.core.transform import Transform
    # create dummy model to scale the average to 0
    class ForwardModel(Transform):
        def __init__(self, in_features=5, out_features=5):
            super().__init__(in_features=5, out_features=5)
            self.mean = 0

        def update_mean(self, x):
            self.mean = torch.mean(x)
        
        def forward(self, x):
            x = x - self.mean
            return x

        def inverse(self, x):
            x = x + self.mean
            return x

    forward_model = ForwardModel()
    inverse_model = Inverse(forward_model)

    input = torch.rand(5)
    forward_model.update_mean(input)
    out = forward_model(input)

    assert(input.mean() == inverse_model(out).mean()) 

if __name__ == "__main__":
    test_inverse()
