import torch

__all__ = ["Stats"]

class Stats(torch.nn.Module):
    """
    Base stats class. 
    To implement a new stats override the compute and forward methods. 
    The parameters of the stats should be set either in the initialization or via the setup_from_datamodule function.
    """
    def setup_from_datamodule(self, datamodule):
        """
        Initialize parameters based on pytorch lighting datamodule.
        """
        pass

    def compute(self, X: torch.Tensor):
        raise NotImplementedError

    def forward(self, X: torch.Tensor):
        raise NotImplementedError

    def teardown(self):
        pass