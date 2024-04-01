import torch

"""
The helper functions for torch. This module is taken from MACE directly:
https://github.com/ACEsuit/mace/blob/main/mace/tools/torch_tools.py
"""

__all__ = ['to_one_hot']


def to_one_hot(indices: torch.Tensor, n_classes: int) -> torch.Tensor:
    """
    Generates one-hot encoding with `n_classes` classes from `indices`

    Parameters
    ----------
    indices: torch.tensor (shape: [N, 1])
        Node incices.
    n_classes: int
        Number of classes.

    Returns
    -------
    encoding: torch.tensor (shape: [N, n_classes])
        The one-hot encoding.
    """
    shape = indices.shape[:-1] + (n_classes,)
    oh = torch.zeros(shape, device=indices.device).view(shape)

    # scatter_ is the in-place version of scatter
    oh.scatter_(dim=-1, index=indices, value=1)

    return oh.view(*shape)


def test_to_one_hot():
    i = torch.tensor([[0], [2], [1]], dtype=torch.int64)
    e = to_one_hot(i, 4)
    assert (
        e == torch.tensor(
            [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0]], dtype=torch.int64
        )
    ).all()


if __name__ == '__main__':
    test_to_one_hot()
