import torch
from mlcolvar.utils._code import broadcast, scatter_mean, scatter_sum


def test_broadcast():
    # Base tensors used across broadcast scenarios.
    src = torch.tensor([1, 2, 3], dtype=torch.float32)
    other = torch.zeros((2, 3, 4), dtype=torch.float32)

    # Case 1: positive `dim` should align src on axis 1 and expand to `other` shape.
    out_pos = broadcast(src, other, dim=1)
    assert out_pos.shape == other.shape
    assert torch.allclose(out_pos[:, :, 0], torch.tensor([[1, 2, 3], [1, 2, 3]], dtype=torch.float32))

    # Case 2: negative `dim` should be resolved consistently with positive indexing.
    out_neg = broadcast(src, other, dim=-2)
    assert out_neg.shape == other.shape
    assert torch.allclose(out_neg[:, :, 1], torch.tensor([[1, 2, 3], [1, 2, 3]], dtype=torch.float32))


def test_scatter_sum():
    # Shared source/index pair: positions 0 and 2 go to bucket 0, position 1 to bucket 1.
    src = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    index = torch.tensor([0, 1, 0], dtype=torch.long)

    # Case 1: standard path creating the output tensor internally.
    out = scatter_sum(src, index, dim=1)
    expected = torch.tensor([[4.0, 2.0], [10.0, 5.0]])
    assert torch.allclose(out, expected)

    # Case 2: same reduction but with preallocated output tensor.
    prealloc = torch.zeros((2, 2), dtype=torch.float32)
    out_prealloc = scatter_sum(src, index, dim=1, out=prealloc)
    assert torch.allclose(out_prealloc, expected)

    # Case 3: empty input should return an empty output on the reduced axis.
    empty_src = torch.tensor([], dtype=torch.float32)
    empty_index = torch.tensor([], dtype=torch.long)
    out_empty = scatter_sum(empty_src, empty_index, dim=0)
    assert out_empty.shape == torch.Size([0])


def test_scatter_mean():
    # Shared index mapping used for both floating-point and integer branches.
    index = torch.tensor([0, 1, 0], dtype=torch.long)

    # Case 1: floating-point tensors use true division.
    src_float = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    out_float = scatter_mean(src_float, index, dim=1)
    expected_float = torch.tensor([[2.0, 2.0], [5.0, 5.0]])
    assert torch.allclose(out_float, expected_float)

    # Case 2: integer tensors use floor division (`rounding_mode='floor'` path).
    src_int = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.int64)
    out_int = scatter_mean(src_int, index, dim=1)
    expected_int = torch.tensor([[2, 2], [5, 5]], dtype=torch.int64)
    assert torch.equal(out_int, expected_int)
