import torch
import numpy as np

"""
The radial functions. This module is taken from MACE directly:
https://github.com/ACEsuit/mace/blob/main/mace/modules/radial.py
"""

__all__ = ['RadialEmbeddingBlock']


class GaussianBasis(torch.nn.Module):
    """
    The Gaussian basis functions.

    Parameters
    ----------
    cutoff: float
        The cutoff radius.
    n_bases: int
        Size of the basis set.
    """
    def __init__(self, cutoff: float, n_bases=32) -> None:
        super().__init__()

        offset = torch.linspace(
            start=0.0,
            end=cutoff,
            steps=n_bases,
            dtype=torch.get_default_dtype(),
        )
        coeff = -0.5 / (offset[1] - offset[0]).item() ** 2
        self.register_buffer(
            'cutoff', torch.tensor(cutoff, dtype=torch.get_default_dtype())
        )
        self.register_buffer(
            'coeff', torch.tensor(coeff, dtype=torch.get_default_dtype())
        )
        self.register_buffer('offset', offset)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dist = x.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))[0]

    def __repr__(self) -> str:
        result = 'GAUSSIANBASIS [ '

        data_string = '\033[32m{:d}\033[0m\033[36m 󰯰 \033[0m'
        result = result + data_string.format(len(self.offset))
        result = result + '| '
        data_string = '\033[32m{:f}\033[0m\033[36m 󰳁 \033[0m'
        result = result + data_string.format(self.cutoff)
        result = result + ']'

        return result


class BesselBasis(torch.nn.Module):
    """
    The Bessel radial basis functions (equation (7) in [1]).

    Parameters
    ----------
    cutoff: float
        The cutoff radius.
    n_bases: int
        Size of the basis set.
    trainable: bool
        If use trainable basis set parameters.

    References
    ----------
    .. [1] Klicpera, J.; Groß, J.; Günnemann, S. Directional Message Passing
    for Molecular Graphs; ICLR 2020.
    """

    def __init__(self, cutoff: float, n_bases=8, trainable=False) -> None:
        super().__init__()

        bessel_weights = (
            np.pi
            / cutoff
            * torch.linspace(
                start=1.0,
                end=n_bases,
                steps=n_bases,
                dtype=torch.get_default_dtype(),
            )
        )
        if trainable:
            self.bessel_weights = torch.nn.Parameter(bessel_weights)
        else:
            self.register_buffer('bessel_weights', bessel_weights)

        self.register_buffer(
            'cutoff', torch.tensor(cutoff, dtype=torch.get_default_dtype())
        )
        self.register_buffer(
            'prefactor',
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
            'p', torch.tensor(p, dtype=torch.get_default_dtype())
        )
        self.register_buffer(
            'cutoff', torch.tensor(cutoff, dtype=torch.get_default_dtype())
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


class RadialEmbeddingBlock(torch.nn.Module):
    """
    The radial embedding block [1].

    Parameters
    ----------
    cutoff: float
        The cutoff radius.
    n_bases: int
        Size of the basis set.
    n_polynomials: bool
        Order of the polynomial.
    basis_type: str
        Type of the basis function.

    References
    ----------
    .. [1] Klicpera, J.; Groß, J.; Günnemann, S. Directional Message Passing
    for Molecular Graphs; ICLR 2020.
    """

    def __init__(
        self,
        cutoff: float,
        n_bases: int = 8,
        n_polynomials: int = 6,
        basis_type: str = 'bessel',
    ) -> None:
        super().__init__()
        self.n_out = n_bases
        if basis_type == 'bessel':
            self.basis_fn = BesselBasis(cutoff=cutoff, n_bases=n_bases)
            self.cutoff_fn = PolynomialCutoff(cutoff=cutoff, p=n_polynomials)
        elif basis_type == 'gaussian':
            self.basis_fn = GaussianBasis(cutoff=cutoff, n_bases=n_bases)
            self.cutoff_fn = None
        else:
            raise RuntimeError(
                'Unknown basis function type "{:s}" !'.format(basis_type)
            )

    def forward(self, edge_lengths: torch.Tensor) -> torch.Tensor:
        """
        The forward pass.

        Parameters
        ----------
        edge_lengths: torch.Tensor (shape: [n_edges, 1])
            Lengths of edges.

        Returns
        -------
        edge_embedding: torch.Tensor (shape: [n_edges, n_bases])
            The radial edge embedding.
        """
        r = self.basis_fn(edge_lengths)  # shape: [n_edges, n_bases]
        if self.cutoff_fn is not None:
            c = self.cutoff_fn(edge_lengths)  # shape: [n_edges, 1]
            return r * c
        else:
            return r


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


def test_gaussian_basis() -> None:
    dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)

    data = torch.tensor([
        [0.9998611207557263, 0.6166385641763439],
        [0.9950124791926823, 0.6669768108584744],
        [0.9833348700493460, 0.7164317992468783],
        [0.9650691177896804, 0.7642281651714904],
        [0.9405880633643421, 0.8095716486678869],
        [0.9103839103891423, 0.8516705072294410],
        [0.8750517756337902, 0.8897581848801761],
        [0.8352702114112720, 0.9231163463866358],
        [0.7917795893122607, 0.9510973184771084],
        [0.7453593045429805, 0.9731449630580510],
    ])

    rbf = GaussianBasis(6.0, 2)

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


def test_radial_embedding_block():
    dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)

    data = torch.tensor([
        [0.302161784075405670,  0.603495363703668900],
        [0.297344780473306900,  0.565583382110980900],
        [0.285645292705329600,  0.479124599728231300],
        [0.266549578182040000,  0.356712961747292670],
        [0.238761404317085600,  0.216790818528859370],
        [0.201179558989195350,  0.083655164534829570],
        [0.154832684273361420, -0.016206633178216297],
        [0.104419964978618930, -0.064535087460860160],
        [0.057909938358517744, -0.063080025890725560],
        [0.023554408472511446, -0.035008673547055544]
    ])

    embedding = RadialEmbeddingBlock(6, 2, 6)

    data_new = torch.stack(
        [embedding(torch.ones(1) * i * 0.5 + 0.1) for i in range(0, 10)]
    )

    assert (torch.abs(data - data_new) < 1E-12).all()

    torch.set_default_dtype(dtype)


if __name__ == '__main__':
    test_bessel_basis()
    test_gaussian_basis()
    test_polynomial_cutoff()
    test_radial_embedding_block()
