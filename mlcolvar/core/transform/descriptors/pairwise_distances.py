import torch

from mlcolvar.core.transform import Transform
from mlcolvar.core.transform.descriptors.utils import compute_distances_matrix

from typing import Union

__all__ = ["PairwiseDistances"]

class PairwiseDistances(Transform):
    """
    Pairwise distances transform, compute all the non duplicated pairwise distances for a set of atoms from their positions
    """

    def __init__(self, 
                 n_atoms : int,
                 PBC: bool,
                 cell: Union[float, list],
                 scaled_coords : bool = False,
                 slicing_pairs : list = None) -> torch.Tensor:
        """Initialize a pairwise distances matrix object.

        Parameters
        ----------
        n_atoms : int
            Number of atoms in the system
        PBC : bool
            Switch for Periodic Boundary Conditions use
        cell : Union[float, list]
            Dimensions of the real cell, orthorombic-like cells only
        scaled_coords : bool
            Switch for coordinates scaled on cell's vectors use, by default False
        slicing_pairs : list
            Indeces of the subset of distances to be returned, by default None

        Returns
        -------
        torch.Tensor
            Non duplicated pairwise distances between all the atoms
        """
        if slicing_pairs is None:
            super().__init__(in_features=int(n_atoms*3), out_features=int(n_atoms*(n_atoms-1) / 2))
        else: 
            super().__init__(in_features=int(n_atoms*3), out_features=len(slicing_pairs))

        # parse args
        self.n_atoms = n_atoms
        self.PBC = PBC
        self.cell = cell
        self.scaled_coords = scaled_coords
        if slicing_pairs is not None:
            self.slicing_pairs = torch.Tensor(slicing_pairs).to(torch.long)
        else:
            self.slicing_pairs = slicing_pairs

    def compute_pairwise_distances(self, pos):
        dist = compute_distances_matrix(pos=pos,
                                        n_atoms=self.n_atoms,
                                        PBC=self.PBC,
                                        cell=self.cell,
                                        scaled_coords=self.scaled_coords)
        batch_size = dist.shape[0]
        if self.slicing_pairs is None:
            device = pos.device
            # mask out diagonal elements
            aux_mask = torch.ones_like(dist, device=device) - torch.eye(dist.shape[-1], device=device)
            # keep upper triangular part to avoid duplicates
            unique = aux_mask.triu().nonzero(as_tuple=True)
            pairwise_distances = dist[unique].reshape((batch_size, -1)) 
            return pairwise_distances
        else:
            return dist[:, self.slicing_pairs[:, 0], self.slicing_pairs[:, 1]]
        

    def forward(self, x: torch.Tensor):
        x = self.compute_pairwise_distances(x)
        return x

def test_pairwise_distances():
    # simple test based on alanine distances
    pos_abs = torch.Tensor([[ 1.4970,  1.3861, -0.0273, -1.4933,  1.5070, -0.1133, -1.4473, -1.4193,
                        -0.0553,  1.4940,  1.4990, -0.2403,  1.4780, -1.4173, -0.3363, -1.4243,
                        -1.4093, -0.4293,  1.3530, -1.4313, -0.4183,  1.3060,  1.4750, -0.4333,
                        1.2970, -1.3233, -0.4643,  1.1670, -1.3253, -0.5354]])
    pos_abs.requires_grad = True

    cell = torch.Tensor([3.0233])
    
    pos_scaled = pos_abs / cell

    ref_distances = torch.Tensor([[0.1521, 0.2335, 0.2412, 0.3798, 0.4733, 0.4649, 0.4575, 0.5741, 0.6815,
                                0.1220, 0.1323, 0.2495, 0.3407, 0.3627, 0.3919, 0.4634, 0.5885, 0.2280,
                                0.2976, 0.3748, 0.4262, 0.4821, 0.5043, 0.6376, 0.1447, 0.2449, 0.2454,
                                0.2705, 0.3597, 0.4833, 0.1528, 0.1502, 0.2370, 0.2408, 0.3805, 0.2472,
                                0.3243, 0.3159, 0.4527, 0.1270, 0.1301, 0.2440, 0.2273, 0.2819, 0.1482]])
    
    # cell = torch.Tensor([1., 2., 1.])
  
    model = PairwiseDistances(n_atoms = 10,
                              PBC = True,
                              cell = cell,
                              scaled_coords = False)
    out = model(pos_abs)
    assert(out.reshape(pos_abs.shape[0], -1).shape[-1] == model.out_features)
    assert(torch.allclose(out, ref_distances, atol=1e-3))
    out.sum().backward()


    model = PairwiseDistances(n_atoms = 10,
                              PBC = True,
                              cell = cell,
                              scaled_coords = False,
                              slicing_pairs=[[0, 1], [0, 2]])
    out = model(pos_abs)
    assert(torch.allclose(out, ref_distances[:, [0, 1]], atol=1e-3))
    out.sum().backward()

    model = PairwiseDistances(n_atoms = 10,
                              PBC = True,
                              cell = cell,
                              scaled_coords = True)
    out = model(pos_scaled)
    assert(out.reshape(pos_scaled.shape[0], -1).shape[-1] == model.out_features)
    assert(torch.allclose(out, ref_distances, atol=1e-3))
    out.sum().backward()


if __name__ == "__main__":
    test_pairwise_distances()