import torch

__all__ = ["Transform"]


class Transform(torch.nn.Module):
    """
    Base transform class.
    To implement a new transform override the forward method.
    The parameters of the transform should be set either in the initialization or via the setup_from_datamodule function.
    """

    def __init__(self, in_features: int, out_features: int):
        """Transform class options.

        Parameters
        ----------
        in_features : int
            Number of inputs of the transform
        out_features : int
            Number of outputs of the transform
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

    def setup_from_datamodule(self, datamodule):
        """
        Initialize parameters based on pytorch lighting datamodule.
        """
        pass

    def forward(self, X: torch.Tensor):
        raise NotImplementedError

    def teardown(self):
        pass
