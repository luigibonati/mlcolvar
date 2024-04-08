import torch
from typing import Tuple

"""
The helper functions for torch. This module is taken from MACE directly:
https://github.com/ACEsuit/mace/blob/main/mace/tools/torch_tools.py
"""

__all__ = [
    'to_one_hot',
    'set_default_dtype',
    'get_edge_vectors_and_lengths'
]


def to_one_hot(indices: torch.Tensor, n_classes: int) -> torch.Tensor:
    """
    Generates one-hot encoding with `n_classes` classes from `indices`

    Parameters
    ----------
    indices: torch.Tensor (shape: [N, 1])
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


def get_edge_vectors_and_lengths(
    positions: torch.Tensor,  # [n_nodes, 3]
    edge_index: torch.Tensor,  # [2, n_edges]
    shifts: torch.Tensor,  # [n_edges, 3]
    normalize: bool = True,
    eps: float = 1e-9,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Calculate edge vectors and lengths by indices and shift vectors.

    Parameters
    ----------
    position: torch.Tensor (shape: [n_atoms, 3])
        The position vector.
    edge_index: torch.Tensor (shape: [2, n_edges])
        The edge indices.
    shifts: torch.Tensor (shape: [n_edges, 3])
        The shift vector.
    normalize: bool
        If return the normalized distance vectors.
    eps: float
        The tolerance of zero-length vectors.

    Returns
    -------
    vectors: torch.Tensor (shape: [n_edges, 3])
        The distance vectors.
    lengths: torch.Tensor (shape: [n_edges, 1])
        The edge lengths.
    """
    sender = edge_index[0]
    receiver = edge_index[1]
    vectors = positions[receiver] - positions[sender] + shifts  # [n_edges, 3]
    lengths = torch.linalg.norm(vectors, dim=-1, keepdim=True)  # [n_edges, 1]
    if normalize:
        vectors_normed = vectors / (lengths + eps)
        return vectors_normed, lengths

    return vectors, lengths


def set_default_dtype(dtype: str) -> None:
    """
    Wrapper function of `torch.set_default_dtype`.

    Parameters
    ----------
    dtype: str
        The data type.
    """
    if not isinstance(dtype, str):
        raise TypeError('A string is required to set TORCH default dtype!')
    dtype = dtype.lower()
    if dtype in ['float', 'float32']:
        torch.set_default_dtype(torch.float32)
    elif dtype in ['double', 'float64']:
        torch.set_default_dtype(torch.float64)
    else:
        raise RuntimeError(
            'Unknown/Unsupported data type: "{:s}"!'.format(dtype)
        )


def test_to_one_hot() -> None:
    i = torch.tensor([[0], [2], [1]], dtype=torch.int64)
    e = to_one_hot(i, 4)
    assert (
        e == torch.tensor(
            [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0]], dtype=torch.int64
        )
    ).all()


def test_get_edge_vectors_and_lengths() -> None:
    dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)

    data = dict()
    data['positions'] = torch.tensor(
        [[0.0, 0.0, 0.0], [0.07, 0.07, 0.0], [0.07, -0.07, 0.0]],
        dtype=torch.float64
    )
    data['edge_index'] = torch.tensor(
        [[0, 0, 1, 1, 2, 2], [2, 1, 0, 2, 1, 0]]
    )
    data['shifts'] = torch.tensor([
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.2, 0.0],
        [0.0, -0.2, 0.0],
        [0.0, 0.0, 0.0],
    ])

    vectors, distances = get_edge_vectors_and_lengths(**data, normalize=False)
    assert (
        torch.abs(
            vectors
            - torch.tensor([
                [0.0700, -0.0700, 0.0000],
                [0.0700,  0.0700, 0.0000],
                [-0.070, -0.0700, 0.0000],
                [0.0000,  0.0600, 0.0000],
                [0.0000, -0.0600, 0.0000],
                [-0.070,  0.0700, 0.0000]
            ])
        ) < 1E-12
    ).all()
    assert (
        torch.abs(
            distances
            - torch.tensor([
                [0.09899494936611666],
                [0.09899494936611666],
                [0.09899494936611666],
                [0.06000000000000000],
                [0.06000000000000000],
                [0.09899494936611666],
            ])
        ) < 1E-12
    ).all()

    vectors, distances = get_edge_vectors_and_lengths(**data, normalize=True)
    assert (
        torch.abs(
            vectors
            - torch.tensor([
                [0.70710677404369050, -0.7071067740436905, 0.0],
                [0.70710677404369050,  0.7071067740436905, 0.0],
                [-0.7071067740436905, -0.7071067740436905, 0.0],
                [0.00000000000000000,  0.9999999833333336, 0.0],
                [0.00000000000000000, -0.9999999833333336, 0.0],
                [-0.7071067740436905,  0.7071067740436905, 0.0],
            ])
        ) < 1E-12
    ).all()
    assert (
        torch.abs(
            distances
            - torch.tensor([
                [0.09899494936611666],
                [0.09899494936611666],
                [0.09899494936611666],
                [0.06000000000000000],
                [0.06000000000000000],
                [0.09899494936611666],
            ])
        ) < 1E-12
    ).all()

    torch.set_default_dtype(dtype)


def test_set_default_dtype() -> None:
    set_default_dtype('float64')
    t = torch.Tensor([1.0])
    assert t.dtype == torch.float64

    set_default_dtype('float32')
    t = torch.Tensor([1.0])
    assert t.dtype == torch.float32

    set_default_dtype('double')
    t = torch.Tensor([1.0])
    assert t.dtype == torch.float64

    set_default_dtype('float')
    t = torch.Tensor([1.0])
    assert t.dtype == torch.float32


if __name__ == '__main__':
    test_to_one_hot()
    test_get_edge_vectors_and_lengths()
    test_set_default_dtype()
