import torch
import numpy as np

"""
The radial functions. This module is taken from MACE directly:
https://github.com/ACEsuit/mace/blob/main/mace/modules/radial.py
"""

__all__ = ['BesselBasis', 'PolynomialCutoff']


class BesselBasis(torch.nn.Module):
    """
    The Bessel radial basis functions (equation (7) in [1]).

    Parameters
    ----------
    cutoff: float
        The cutoff radius.
    n_basis: int
        Size of the basis set.
    trainable: bool
        If use trainable basis set parameters.

    References
    ----------
    .. [1] Klicpera, J.; Groß, J.; Günnemann, S. Directional Message Passing
    for Molecular Graphs; ICLR 2020.
    """

    def __init__(self, cutoff: float, n_basis=8, trainable=False) -> None:
        super().__init__()

        bessel_weights = (
            np.pi
            / cutoff
            * torch.linspace(
                start=1.0,
                end=n_basis,
                steps=n_basis,
                dtype=torch.get_default_dtype(),
            )
        )
        if trainable:
            self.bessel_weights = torch.nn.Parameter(bessel_weights)
        else:
            self.register_buffer("bessel_weights", bessel_weights)

        self.register_buffer(
            "cutoff", torch.tensor(cutoff, dtype=torch.get_default_dtype())
        )
        self.register_buffer(
            "prefactor",
            torch.tensor(
                np.sqrt(2.0 / cutoff), dtype=torch.get_default_dtype()
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        numerator = torch.sin(self.bessel_weights * x)
        return self.prefactor * (numerator / x)

    def __repr__(self) -> str:
        result = 'BESSELBASIS [ '

        data_string = '\033[32m{:d}\033[0m\033[36m 󰯰 \033[0m'
        result = result + data_string.format(len(self.bessel_weights))
        result = result + '| '
        data_string = '\033[32m{:f}\033[0m\033[36m 󰳁 \033[0m'
        result = result + data_string.format(self.cutoff)
        if self.bessel_weights.requires_grad:
            result = result + '|\033[36m TRAINABLE \033[0m'
        result = result + ']'

        return result


class PolynomialCutoff(torch.nn.Module):
    """
    The Continuous cutoff function (equation (8) in [1]).

    Parameters
    ----------
    cutoff: float
        The cutoff radius.
    p: int
        Order of the polynomial.

    References
    ----------
    .. [1] Klicpera, J.; Groß, J.; Günnemann, S. Directional Message Passing
           for Molecular Graphs; ICLR 2020.
    """
    p: torch.Tensor
    cutoff: torch.Tensor

    def __init__(self, cutoff: float, p: int = 6) -> None:
        super().__init__()
        self.register_buffer(
            "p", torch.tensor(p, dtype=torch.get_default_dtype())
        )
        self.register_buffer(
            "cutoff", torch.tensor(cutoff, dtype=torch.get_default_dtype())
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # fmt: off
        envelope = (
                1.0
                - (self.p + 1.0) * (self.p + 2.0) / 2.0
                * torch.pow(x / self.cutoff, self.p)
                + self.p * (self.p + 2.0)
                * torch.pow(x / self.cutoff, self.p + 1)
                - self.p * (self.p + 1.0) / 2
                * torch.pow(x / self.cutoff, self.p + 2)
        )
        # fmt: on

        # noinspection PyUnresolvedReferences
        return envelope * (x < self.cutoff)

    def __repr__(self) -> str:
        result = 'POLYNOMIALCUTOFF [ '

        data_string = '\033[32m{:d}\033[0m\033[36m 󰰚 \033[0m'
        result = result + data_string.format(int(self.p))
        result = result + '| '
        data_string = '\033[32m{:f}\033[0m\033[36m 󰳁 \033[0m'
        result = result + data_string.format(self.cutoff)
        result = result + ']'

        return result


def test_bessel_basis() -> None:
    dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)

    data = torch.tensor([
        [0.30216178425160090,  0.603495364055576400],
        [0.29735174147757487,  0.565596622727919000],
        [0.28586135770645804,  0.479487014442650350],
        [0.26815929064765680,  0.358867177503655900],
        [0.24496326504279375,  0.222421990229218020],
        [0.21720530022724968,  0.090319042449653110],
        [0.18598678410040770, -0.019467592388889482],
        [0.15252575991598738, -0.094266103787986490],
        [0.11809918979627002, -0.128642857533393970],
        [0.08398320341397922, -0.124823366088228150]
    ])

    rbf = BesselBasis(6.0, 2)

    data_new = torch.stack(
        [rbf(torch.ones(1) * i * 0.5 + 0.1) for i in range(0, 10)]
    )

    assert (torch.abs(data - data_new) < 1E-12).all()

    torch.set_default_dtype(dtype)


def test_polynomial_cutoff() -> None:
    dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)

    data = torch.tensor([
        [1.0000000000000000],
        [0.9999919136092714],
        [0.9995588277320531],
        [0.9957733154296875],
        [0.9803383630544124],
        [0.9390599059360889],
        [0.8554687500000000],
        [0.7184512221655127],
        [0.5317786922725198],
        [0.3214569091796875]
    ])

    cutoff_function = PolynomialCutoff(6.0)

    data_new = torch.stack(
        [cutoff_function(torch.ones(1) * i * 0.5) for i in range(0, 10)]
    )

    assert (torch.abs(data - data_new) < 1E-12).all()

    torch.set_default_dtype(dtype)


if __name__ == '__main__':
    test_bessel_basis()
    test_polynomial_cutoff()
