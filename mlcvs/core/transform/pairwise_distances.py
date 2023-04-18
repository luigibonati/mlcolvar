import torch

from mlcolvar.core.transform import Transform
from mlcolvar.core.transform.utils import compute_distances_matrix

from typing import Union

__all__ = ["PairwiseDistances"]

class PairwiseDistances(Transform):
    """
    Pairwise distances transform, compute all the non duplicated pairwise distances for a set of atoms from their positions
    """

    def __init__(self, 
                 n_atoms : int,
                 PBC: bool,
                 real_cell: Union[float, list],
                 scaled_coords : bool) -> torch.Tensor:
        """Initialize a pairwise distances matrix object.

        Parameters
        ----------
        n_atoms : int
            Number of atoms in the system
        PBC : bool
            Switch for Periodic Boundary Conditions use
        real_cell : Union[float, list]
            Dimensions of the real cell, orthorombic-like cells only
        scaled_coords : bool
            Switch for coordinates scaled on cell's vectors use

        Returns
        -------
        torch.Tensor
            Non duplicated pairwise distances between all the atoms
        """
        super().__init__(in_features=int(n_atoms*3), out_features=int(n_atoms*(n_atoms-1) / 2))

        # parse args
        self.n_atoms = n_atoms
        self.PBC = PBC
        self.real_cell = real_cell
        self.scaled_coords = scaled_coords

    def compute_pairwise_distances(self, pos):
        dist = compute_distances_matrix(pos=pos,
                                        n_atoms=self.n_atoms,
                                        PBC=self.PBC,
                                        real_cell=self.real_cell,
                                        scaled_coords=self.scaled_coords)
        batch_size = dist.shape[0]
        # mask out diagonal elements
        aux_mask = torch.ones_like(dist) - torch.eye(dist.shape[-1])
        # keep upper triangular part to avoid duplicates
        unique = aux_mask.triu().nonzero(as_tuple=True)
        pairwise_distances = dist[unique].reshape((batch_size, -1)) 
        return pairwise_distances

    def forward(self, x: torch.Tensor):
        x = self.compute_pairwise_distances(x)
        return x

def test_pairwise_distances():

    pos = torch.Tensor([ [ [0., 0., 0.],
                           [1., 1., 1.],
                           [1., 1., 1.1] ],
                         [ [0., 0., 0.],
                           [1., 1.1, 1.],
                           [1., 1., 1.] ] ]
                      )
    
    real_cell = torch.Tensor([1., 2., 1.])
    cutoff = 1.8
  
    model = PairwiseDistances(n_atoms = 3,
                              PBC = True,
                              real_cell = real_cell,
                              scaled_coords = False)
    out = model(pos)
    assert(out.reshape(pos.shape[0], -1).shape[-1] == model.out_features)

if __name__ == "__main__":
    test_pairwise_distances()