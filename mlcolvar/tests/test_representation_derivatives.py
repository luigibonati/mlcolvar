import pytest
import torch

from mlcolvar.representation.preparation import JacobianTransform


def test_jacobian_transform():
    jacobian = torch.tensor([
        [[1.0, 0.0], [0.0, 2.0], [1.0, 1.0]],
        [[2.0, 0.0], [0.0, 3.0], [1.0, 2.0]],
    ])
    gradient = torch.tensor([
        [1.0, 2.0],
        [3.0, 4.0],
    ])

    output = JacobianTransform(jacobian)(
        gradient,
        torch.tensor([0, 1]),
    )

    expected = torch.tensor([
        [1.0, 4.0, 3.0],
        [6.0, 12.0, 11.0],
    ])
    torch.testing.assert_close(output, expected)


def test_jacobian_transform_reference_indices():
    jacobian = torch.tensor([
        [[1.0, 0.0], [0.0, 1.0]],
        [[2.0, 0.0], [0.0, 2.0]],
    ])

    output = JacobianTransform(jacobian)(
        torch.ones(2, 2),
        torch.tensor([1, 0]),
    )

    expected = torch.tensor([
        [2.0, 2.0],
        [1.0, 1.0],
    ])
    torch.testing.assert_close(output, expected)


def test_jacobian_transform_requires_ref_idx():
    transform = JacobianTransform(torch.ones(2, 3, 2))

    with pytest.raises(ValueError, match="`ref_idx` is required"):
        transform(torch.ones(2, 2))


def test_jacobian_transform_invalid_ref_idx():
    transform = JacobianTransform(torch.ones(2, 3, 2))

    with pytest.raises(IndexError, match="Invalid Jacobian index"):
        transform(
            torch.ones(1, 2),
            torch.tensor([2]),
        )