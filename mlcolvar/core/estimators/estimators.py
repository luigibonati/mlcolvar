import torch

__all__ = ["Estimator"]


class Estimator(torch.nn.Module):
    """
    Base Estimator class.
    To implement a new estimator override the compute and forward methods.
    The parameters of the estimator should be set either in the initialization or via the setup_from_datamodule function.
    """

    def compute(self, X: torch.Tensor):
        """
        Compute the parameters of the estimator
        """
        raise NotImplementedError

    def forward(self, X: torch.Tensor):
        """
        Apply estimator
        """
        raise NotImplementedError

    def teardown(self):
        pass
