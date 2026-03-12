import torch
import numpy as np

"""
The radial functions. This module is taken from MACE directly:
https://github.com/ACEsuit/mace/blob/main/mace/modules/radial.py
"""

__all__ = ['RadialEmbeddingBlock']


class GaussianBasis(torch.nn.Module):
    """
    Gaussian basis functions.
    """
    def __init__(self, 
                 cutoff: float, 
                 long_range_cutoff: float = -1.0, 
                 n_bases=32) -> None:
        """Initialize a Gaussian basis function

        Parameters
        ----------
        cutoff : float
            Cutoff radius of the basis set
        n_bases : int, optional
            Size of the basis set, by default 32
        long_range_cutoff: float
            Long range cutoff for interaction between subsystem atoms, not used if negative, by default -1.0
        """
        super().__init__()

        offset = torch.linspace(
            start=0.0,
            end=cutoff,
            steps=n_bases,
            dtype=torch.get_default_dtype(),
        )
        coeff = -0.5 / (offset[1] - offset[0]).item() ** 2

        self.register_buffer(
            'coeff', torch.tensor(coeff, dtype=torch.get_default_dtype())
        )
        self.register_buffer('offset', offset)

        if long_range_cutoff > 0:
            offset_lr = torch.linspace(
                start=0.0,
                end=long_range_cutoff,
                steps=n_bases,
                dtype=torch.get_default_dtype(),
            )
            coeff_lr = -0.5 / (offset_lr[1] - offset_lr[0]).item() ** 2
            self.register_buffer(
                'coeff_lr',
                torch.tensor(coeff_lr, dtype=torch.get_default_dtype())
            )
            self.register_buffer('offset_lr', offset_lr)
        else:
            self.register_buffer('coeff_lr', torch.zeros((1, 1)))
            self.register_buffer('offset_lr', torch.zeros((1, 1)))

        self.register_buffer(
            'cutoff',
            torch.tensor(cutoff, dtype=torch.get_default_dtype())
        )
        self.register_buffer(
            'long_range_cutoff',
            torch.tensor(long_range_cutoff, dtype=torch.get_default_dtype())
        )

    def forward(
        self, x: torch.Tensor, edge_masks_lr: torch.Tensor = None
    ) -> torch.Tensor:

        x = x.view(-1, 1)

        dist = x - self.offset.view(1, -1)
        values = torch.exp(self.coeff * dist.pow(2))

        if edge_masks_lr is None:
            return values

        dist_l = x - self.offset_lr.view(1, -1)
        values_l = torch.exp(self.coeff_lr * dist_l.pow(2))

        mask = edge_masks_lr.view(-1, 1)
        return torch.where(mask, values_l, values)

    def __repr__(self) -> str:
        result = 'GAUSSIANBASIS [ '

        data_string = '\033[32m{:d}\033[0m\033[36m 󰯰 \033[0m'
        result = result + data_string.format(len(self.offset))
        result = result + '| '
        data_string = '\033[32m{:f}\033[0m\033[36m 󰳁 \033[0m'
        result = result + data_string.format(self.cutoff)
        if self.long_range_cutoff > 0:
            data_string = '\033[32m{:f}\033[0m\033[36m 󰳁 \033[0m'
            result = result + data_string.format(self.long_range_cutoff)
        result = result + ']'

        return result


class BesselBasis(torch.nn.Module):
    r"""
    Bessel radial basis functions (equation (7) in [1])

    .. math:: RBF_n(d) = \sqrt{\frac{2}{c}\frac{sin(\frac{n\pi}{c}d)}{d}}

    References
    ----------
    .. [1] Gasteiger, J.; Groß, J.; Günnemann, S. Directional Message Passing
    for Molecular Graphs; ICLR 2020.
    """

    def __init__(
            self, 
            cutoff: float, 
            long_range_cutoff: float = -1.0, 
            n_bases=8, 
            trainable=False
        ) -> None:
        """Initializes Bessel radial basis function

        Parameters
        ----------
        cutoff: float
            Cutoff radius of the basis set
        long_range_cutoff: float
            Long range cutoff for interaction between subsystem atoms, not used if negative, by default -1.0
        n_bases: int
            Size of the basis set, by default 8
        trainable: bool
            If to use trainable basis set parameters
        """
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
            'prefactor',
            torch.tensor(
                np.sqrt(2.0 / cutoff), dtype=torch.get_default_dtype()
            )
        )

        if long_range_cutoff > 0:
            bessel_weights_lr = (
                np.pi
                / long_range_cutoff
                * torch.linspace(
                    start=1.0,
                    end=n_bases,
                    steps=n_bases,
                    dtype=torch.get_default_dtype(),
                )
            )
            if trainable:
                self.bessel_weights_lr = torch.nn.Parameter(bessel_weights_lr)
            else:
                self.register_buffer('bessel_weights_lr', bessel_weights_lr)

            self.register_buffer(
                'prefactor_lr',
                torch.tensor(
                    np.sqrt(2.0 / long_range_cutoff), dtype=torch.get_default_dtype()
                )
            )
        else:
            self.register_buffer('bessel_weights_lr', torch.zeros(1))
            self.register_buffer('prefactor_lr', torch.zeros(1))

        self.register_buffer(
            'cutoff', torch.tensor(cutoff, dtype=torch.get_default_dtype())
        )
        self.register_buffer(
            'long_range_cutoff',
            torch.tensor(long_range_cutoff, dtype=torch.get_default_dtype())
        )

    def forward(self, 
                x: torch.Tensor, 
                edge_masks_lr: torch.Tensor = None
               ) -> torch.Tensor:
        numerator = torch.sin(self.bessel_weights * x)
        values = self.prefactor * (numerator / x)

        if edge_masks_lr is not None:

            numerator_lr = torch.sin(self.bessel_weights_lr * x)
            values_lr = self.prefactor_lr * (numerator_lr / x)

            mask = edge_masks_lr.view(-1,1)
            values = torch.where(mask, values_lr, values)

        return values

    def __repr__(self) -> str:
        result = 'BESSELBASIS [ '

        data_string = '\033[32m{:d}\033[0m\033[36m 󰯰 \033[0m'
        result = result + data_string.format(len(self.bessel_weights))
        result = result + '| '
        data_string = '\033[32m{:f}\033[0m\033[36m 󰳁 \033[0m'
        result = result + data_string.format(self.cutoff)
        if self.long_range_cutoff > 0:
            data_string = '\033[32m{:f}\033[0m\033[36m 󰳁 \033[0m'
            result = result + data_string.format(self.long_range_cutoff)
        if self.bessel_weights.requires_grad:
            result = result + '|\033[36m TRAINABLE \033[0m'
        result = result + ']'

        return result


class PolynomialCutoff(torch.nn.Module):
    r"""Continuous cutoff function (equation (8) in [1])

    .. math:: u(d) = 1 - \frac{(p+1)(p+2)}{2}d^p + p(p+2)d^{p+1} - \frac{p(p+1)}{2}d^{p+2}

    References
    ----------
    .. [1] Gasteiger, J.; Groß, J.; Günnemann, S. Directional Message Passing
           for Molecular Graphs; ICLR 2020.
    """
    p: torch.Tensor
    cutoff: torch.Tensor
    long_range_cutoff: torch.Tensor

    def __init__(self, 
                 cutoff: float, 
                 long_range_cutoff: float = -1.0, 
                 p: int = 6
                ) -> None:
        """Initializes a polynomial cutoff function.

        Parameters
        ----------
        cutoff: float
            The cutoff radius.
        long_range_cutoff: float
            Long range cutoff for interaction between subsystem atoms, not used if negative, by default -1.0
        p: int
            Order of the polynomial, by default 6
        """
        super().__init__()
        self.register_buffer(
            'p', torch.tensor(p, dtype=torch.get_default_dtype())
        )
        self.register_buffer(
            'cutoff', torch.tensor(cutoff, dtype=torch.get_default_dtype())
        )
        self.register_buffer(
            'long_range_cutoff', torch.tensor(long_range_cutoff, dtype=torch.get_default_dtype())
        )

    def forward(self, 
                x: torch.Tensor, 
                edge_masks_lr: torch.Tensor = None
                ) -> torch.Tensor:
        if edge_masks_lr is None:
            c = self.cutoff
        else:
            c = self.cutoff * ~edge_masks_lr + self.long_range_cutoff * edge_masks_lr
        # fmt: off
        envelope = (
            1.0
            - (self.p + 1.0) * (self.p + 2.0) / 2.0
            * torch.pow(x / c, self.p)
            + self.p * (self.p + 2.0)
            * torch.pow(x / c, self.p + 1)
            - self.p * (self.p + 1.0) / 2
            * torch.pow(x / c, self.p + 2)
        )
        # fmt: on

        # noinspection PyUnresolvedReferences
        return envelope * (x < c)

    def __repr__(self) -> str:
        result = 'POLYNOMIALCUTOFF [ '

        data_string = '\033[32m{:d}\033[0m\033[36m 󰰚 \033[0m'
        result = result + data_string.format(int(self.p))
        result = result + '| '
        data_string = '\033[32m{:f}\033[0m\033[36m 󰳁 \033[0m'
        result = result + data_string.format(self.cutoff)
        if self.long_range_cutoff > 0:
            data_string = '\033[32m{:f}\033[0m\033[36m 󰳁 \033[0m'
            result = result + data_string.format(self.long_range_cutoff)
        result = result + ']'

        return result


class RadialEmbeddingBlock(torch.nn.Module):
    """
    Radial embedding block [1]

    References
    ----------
    .. [1] Gasteiger, J.; Groß, J.; Günnemann, S. Directional Message Passing
    for Molecular Graphs; ICLR 2020.
    """

    def __init__(
        self,
        cutoff: float,
        long_range_cutoff: float = -1.0,
        n_bases: int = 8,
        n_polynomials: int = 6,
        basis_type: str = 'bessel',
    ) -> None:
        """Initializes a radial embedding block

        Parameters
        ----------
        cutoff : float
            Cutoff radius.
        long_range_cutoff: float
            Long range cutoff for interaction between subsystem atoms, not used if negative, by default -1.0
        n_bases : int, optional
            Size of the basis set, by default 8
        n_polynomials : int, optional
            Order of the polynomial for the polynomial cutoff, by default 6
        basis_type : str, optional
            Type fo the basis function, by default 'bessel'

        Raises
        ------
        RuntimeError
            _description_
        """
        super().__init__()
        self.n_out = n_bases
        if basis_type == 'bessel':
            self.bessel_fn = BesselBasis(
                cutoff=cutoff, long_range_cutoff=long_range_cutoff, n_bases=n_bases
            )
        elif basis_type == 'gaussian':
            self.bessel_fn = GaussianBasis(
                cutoff=cutoff, long_range_cutoff=long_range_cutoff, n_bases=n_bases
            )
        else:
            raise RuntimeError(
                'Unknown basis function type "{:s}" !'.format(basis_type)
            )
        if n_polynomials > 0:
            self.cutoff_fn = PolynomialCutoff(
                cutoff=cutoff, long_range_cutoff=long_range_cutoff, p=n_polynomials
            )
        else:
            self.cutoff_fn = None

    def forward(
        self,
        edge_lengths: torch.Tensor,
        edge_masks_lr: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        The forward pass of RadialEmbeddingBlock

        Parameters
        ----------
        edge_lengths: torch.Tensor (shape: [n_edges, 1])
            Lengths of edges.
        edge_masks_lr:  torch.Tensor (shape: [1, n_edges])
            Mask for long range edges.

        Returns
        -------
        edge_embedding: torch.Tensor (shape: [n_edges, n_bases])
            The radial edge embedding.
        """
        r = self.bessel_fn(edge_lengths, edge_masks_lr)
        if self.cutoff_fn is not None:
            c = self.cutoff_fn(edge_lengths, edge_masks_lr)
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

    rbf = BesselBasis(6.0, n_bases=2)

    data_new = rbf(
        torch.tensor([i * 0.5 + 0.1 for i in range(0, 10)]).view(-1, 1)
    )

    assert (torch.abs(data - data_new) < 1E-12).all()

    rbf = BesselBasis(6.0, long_range_cutoff=10.0, n_bases=2)

    data_new = rbf(
        torch.tensor([i * 0.5 + 0.1 for i in range(0, 10)]).view(-1, 1)
    )

    assert (torch.abs(data - data_new) < 1E-12).all()

    data_1 = torch.tensor([
        [0.14047318504712697, 0.280807740020145600],
        [0.29735174147757487, 0.565596622727919000],
        [0.13771654840461342, 0.259149703921308900],
        [0.26815929064765680, 0.358867177503655900],
        [0.13052398436734916, 0.206268360966214370],
        [0.21720530022724968, 0.090319042449653110],
        [0.11931667012564413, 0.134131833956580900],
        [0.15252575991598738, -0.09426610378798649],
        [0.10474546144085174, 0.058446104279945380],
        [0.08398320341397922, -0.12482336608822815],
    ])

    index = torch.tensor([True, False] * 5)
    data_new = rbf(
        torch.tensor([i * 0.5 + 0.1 for i in range(0, 10)]).view(-1, 1),
        index.view(-1, 1)
    )

    assert (torch.abs(data[~index, :] - data_new[~index, :]) < 1E-12).all()
    assert (torch.abs(data_1[index, :] - data_new[index, :]) < 1E-12).all()

    torch.set_default_dtype(dtype)

    print(rbf)


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
        [0.7453593045429805, 0.9731449630580510]
    ])

    rbf = GaussianBasis(6.0, n_bases=2)

    data_new = rbf(
        torch.tensor([i * 0.5 + 0.1 for i in range(0, 10)]).view(-1, 1)
    )

    assert (torch.abs(data - data_new) < 1E-12).all()

    rbf = GaussianBasis(6.0, long_range_cutoff=60.0, n_bases=2)

    index = torch.tensor([True, False] * 5)

    data_new = rbf(
        torch.tensor([i * 0.5 + 0.1 for i in range(0, 10)]).view(-1, 1),
        index.view(-1, 1)
    )
    assert (torch.abs(data[~index, :] - data_new[~index, :]) < 1E-12).all()

    data_new = rbf(
        torch.tensor([i * 0.5 + 0.1 for i in range(0, 10)]).view(-1, 1) * 10,
        index.view(-1, 1)
    )

    assert (torch.abs(data[index, :] - data_new[index, :]) < 1E-12).all()

    torch.set_default_dtype(dtype)

    print(rbf)


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

    data_new = cutoff_function(
        torch.tensor([i * 0.5 for i in range(0, 10)]).view(-1, 1)
    )

    assert (torch.abs(data - data_new) < 1E-12).all()

    cutoff_function = PolynomialCutoff(6.0, 60.0)

    index = torch.tensor([True, False] * 5)
    data_new = cutoff_function(
        torch.tensor([i * 0.5 for i in range(0, 10)]).view(-1, 1),
        index.view(-1, 1)
    )

    assert (torch.abs(data[~index] - data_new[~index]) < 1E-12).all()

    data_new = cutoff_function(
        torch.tensor([i * 0.5 for i in range(0, 10)]).view(-1, 1) * 10,
        index.view(-1, 1)
    )

    assert (data_new[~index][2:] == 0).all()
    assert (torch.abs(data[index] - data_new[index]) < 1E-12).all()

    torch.set_default_dtype(dtype)

    print(cutoff_function)

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

    embedding = RadialEmbeddingBlock(6, -1.0, 2, 6)

    data_new = embedding(
        torch.tensor([i * 0.5 + 0.1 for i in range(0, 10)]).view(-1, 1)
    )

    assert (torch.abs(data - data_new) < 1E-12).all()

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
        [0.7453593045429805, 0.9731449630580510]
    ])

    embedding = RadialEmbeddingBlock(6, 60, 2, 0, 'gaussian')

    index = torch.tensor([True, False] * 5)
    data_new = embedding(
        torch.tensor([i * 0.5 + 0.1 for i in range(0, 10)]).view(-1, 1),
        index.view(-1, 1)
    )
    assert (torch.abs(data[~index, :] - data_new[~index, :]) < 1E-12).all()

    data_new = embedding(
        torch.tensor([i * 0.5 + 0.1 for i in range(0, 10)]).view(-1, 1) * 10,
        index.view(-1, 1)
    )

    assert (torch.abs(data[index, :] - data_new[index, :]) < 1E-12).all()

    torch.set_default_dtype(dtype)
