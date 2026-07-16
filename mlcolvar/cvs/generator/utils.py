import torch
import lightning
import torch_geometric
from typing import Union, Tuple
from mlcolvar.cvs import BaseCV
from mlcolvar.core import FeedForward
from mlcolvar.core.loss.utils.smart_derivatives import SmartDerivatives
from mlcolvar.data import DictDataset
from mlcolvar.utils._code import scatter_sum
import numpy as np

from mlcolvar.data import DictDataset
import gc

class SoftmaxPostProcessing(torch.nn.Module):
    """Apply a softmax normalization followed by a learnable linear mixing.

    Parameters
    ----------
    r : int, default=4
        Number of representation channels. This is both the input and output
        dimension of the final linear layer.
    """

    def __init__(self, r: int = 4):
        super().__init__()
        self.final_linear = torch.nn.Linear(r, r)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Normalize the last dimension and apply the final linear layer.

        Parameters
        ----------
        input : torch.Tensor
            Tensor of shape ``(..., r)``.

        Returns
        -------
        torch.Tensor
            Tensor of shape ``(..., r)`` after softmax normalization and
            linear projection.
        """
        input = torch.nn.functional.softmax(input, dim=-1)
        return self.final_linear(input)








