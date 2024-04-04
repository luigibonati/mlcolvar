import torch

from mlcolvar.graph.core.nn import radial

"""
The GNN building blocks. Some of the blocks are taken from MACE directly:
https://github.com/ACEsuit/mace/blob/main/mace/modules/blocks.py
"""

__all__ = ['RadialEmbeddingBlock']


class RadialEmbeddingBlock(torch.nn.Module):
    """
    The radial embedding block using a Bessel basis set and a smooth cutoff
    function [1].

    Parameters
    ----------
    cutoff: float
        The cutoff radius.
    n_basis: int
        Size of the basis set.
    n_polynomial: bool
        Order of the polynomial.

    References
    ----------
    .. [1] Klicpera, J.; Groß, J.; Günnemann, S. Directional Message Passing
    for Molecular Graphs; ICLR 2020.
    """

    def __init__(
        self,
        cutoff: float,
        n_basis: int = 8,
        n_polynomial: int = 6,
    ) -> None:
        super().__init__()
        self.n_out = n_basis
        self.bessel_fn = radial.BesselBasis(cutoff=cutoff, n_basis=n_basis)
        self.cutoff_fn = radial.PolynomialCutoff(cutoff=cutoff, p=n_polynomial)

    def forward(self, edge_lengths: torch.Tensor) -> torch.Tensor:
        """
        The forward pass.

        Parameters
        ----------
        edge_lengths: torch.Tensor (shape: [n_edges, 1])
            Lengths of edges.

        Returns
        -------
        edge_embedding: torch.Tensor (shape: [n_edges, n_basis])
            The radial edge embedding.
        """
        r = self.bessel_fn(edge_lengths)  # shape: [n_edges, n_basis]
        c = self.cutoff_fn(edge_lengths)  # shape: [n_edges, 1]
        return r * c


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
    test_radial_embedding_block()
