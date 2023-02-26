import torch

__all__ = ["Transform"]

class Transform(torch.nn.Module):
    """
    Base transform class. 
    To implement a new transform override the forward method. 
    The parameters of the transform should be set either in the initialization or via the setup_from_datamodule function.
    """

    def setup_from_datamodule(self, datamodule):
        """
        Initialize parameters based on pytorch lighting datamodule.
        """
        pass

    def forward(self, X: torch.Tensor):
        raise NotImplementedError

    def teardown(self):
        pass