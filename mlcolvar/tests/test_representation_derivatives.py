import pytest
import torch

from mlcolvar.representation.derivatives import (
    CachedRepresentationDerivatives,
)


def test_cached_representation_derivatives():
    jacobian = torch.tensor([
        [
            [1.0, 0.0],
            [0.0, 2.0],
            [1.0, 1.0],
        ],
        [
            [2.0, 0.0],
            [0.0, 3.0],
            [1.0, 2.0],
        ],
    ])

    derivatives = CachedRepresentationDerivatives(
        jacobian
    )

    gradient_latent = torch.tensor([
        [1.0, 2.0],
        [3.0, 4.0],
    ])

    output = derivatives(
        gradient_latent,
        torch.tensor([0, 1]),
    )

    expected = torch.tensor([
        [1.0, 4.0, 3.0],
        [6.0, 12.0, 11.0],
    ])

    torch.testing.assert_close(
        output,
        expected,
    )


def test_cached_representation_derivatives_requires_ref_idx():
    derivatives = CachedRepresentationDerivatives(
        torch.ones(2, 3, 2)
    )

    with pytest.raises(
        ValueError,
        match="`ref_idx` is required",
    ):
        derivatives(
            torch.ones(2, 2)
        )


def test_cached_representation_derivatives_invalid_ref_idx():
    derivatives = CachedRepresentationDerivatives(
        torch.ones(2, 3, 2)
    )

    with pytest.raises(
        IndexError,
        match="Invalid cached derivative index",
    ):
        derivatives(
            torch.ones(1, 2),
            torch.tensor([2]),
        )