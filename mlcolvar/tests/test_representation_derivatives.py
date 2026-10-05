import pytest
import torch
from torch import nn

from mlcolvar.data import DictDataset
from mlcolvar.representation.preparation import JacobianTransform, compute_jacobian


def test_compute_jacobian():
    model = nn.Linear(3, 2, bias=False)
    dataset = DictDataset({"data": torch.randn(4, 3)})
    jacobian = compute_jacobian(
        model, dataset, indices=torch.tensor([1, 3]), batch_size=1
    )
    expected = model.weight.detach().T.unsqueeze(0).expand(2, -1, -1)
    torch.testing.assert_close(jacobian, expected)


@pytest.mark.parametrize(
    ("ref_idx", "expected"),
    [
        (
            torch.tensor([0, 1]),
            torch.tensor([[1., 4., 3.], [6., 12., 11.]]),
        ),
        (
            torch.tensor([1, 0]),
            torch.tensor([[2., 6., 5.], [3., 8., 7.]]),
        ),
    ],
)
def test_jacobian_transform(ref_idx, expected):
    jacobian = torch.tensor([
        [[1., 0.], [0., 2.], [1., 1.]],
        [[2., 0.], [0., 3.], [1., 2.]],
    ])
    gradient = torch.tensor([[1., 2.], [3., 4.]])
    output = JacobianTransform(jacobian)(gradient, ref_idx)
    torch.testing.assert_close(output, expected)


@pytest.mark.parametrize(
    ("ref_idx", "error"),
    [
        (None, ValueError),
        (torch.tensor([-1]), IndexError),
        (torch.tensor([2]), IndexError),
    ],
)
def test_jacobian_transform_errors(ref_idx, error):
    transform = JacobianTransform(torch.ones(2, 3, 2))
    with pytest.raises(error):
        transform(torch.ones(1, 2), ref_idx)