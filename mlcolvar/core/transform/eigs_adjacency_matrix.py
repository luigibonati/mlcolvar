import torch

from mlcolvar.core.transform import Transform
from mlcolvar.core.transform.utils import compute_adjacency_matrix

from typing import Union

__all__ = ["EigsAdjMat"]

class EigsAdjMat(Transform):
    """
    Eigenvalues of adjacency matrix transform, compute the eigenvalues of the adjacency matrix for a set of atoms from their positions
    """

    def __init__(self,
                 mode : str,
                 cutoff : float, 
                 n_atoms : int,
                 PBC: bool,
                 real_cell: Union[float, list],
                 scaled_coords : bool,
                 switching_function = None) -> torch.Tensor:
        """Initialize an eigenvalues of an adjacency matrix object.

        Parameters
        ----------
        mode : str
            Mode for cutoff application, either:
            - 'continuous': applies a switching function to the distances which can be specified with switching_function keyword, has stable derivatives
            - 'discontinuous': set at zero everything above the cutoff and one below, derivatives may be be incorrect
        cutoff : float
            Cutoff for the adjacency criterion 
        n_atoms : int
            Number of atoms in the system
        PBC : bool
            Switch for Periodic Boundary Conditions use
        real_cell : Union[float, list]
            Dimensions of the real cell, orthorombic-like cells only
        scaled_coords : bool
            Switch for coordinates scaled on cell's vectors use
        switching_function : _type_, optional
            Switching function to be applied for the cutoff, can be either initialized as a switching_functions/SwitchingFunctions class or a simple function, by default None

        Returns
        -------
        torch.Tensor
            Adjacency matrix of all the n_atoms according to cutoff
        """
        super().__init__(in_features=int(n_atoms*3), out_features=n_atoms)

        # parse args
        self.mode = mode
        self.cutoff = cutoff 
        self.n_atoms = n_atoms
        self.PBC = PBC
        self.real_cell = real_cell
        self.scaled_coords = scaled_coords
        self.switching_function = switching_function

    def compute_adjacency_matrix(self, x):
        x = compute_adjacency_matrix(pos=x,
                                     mode = self.mode,
                                     cutoff = self.cutoff, 
                                     n_atoms = self.n_atoms,
                                     PBC = self.PBC,
                                     real_cell = self.real_cell,
                                     scaled_coords = self.scaled_coords,
                                     switching_function=self.switching_function)
        return x
    
    def get_eigenvalues(self, x):
        eigs = torch.linalg.eigvalsh(x)
        return eigs

    def forward(self, x: torch.Tensor):
        x = self.compute_adjacency_matrix(x)
        eigs = self.get_eigenvalues(x)
        return eigs

def test_eigs_of_adj_matrix():
    from mlcolvar.core.transform.switching_functions import SwitchingFunctions
    
    n_atoms=2
    pos = torch.Tensor([ [ [0., 0., 0.],
                           [1., 1., 1.] ],
                         [ [0., 0., 0.],
                           [1., 1.1, 1.] ] ]
                      )
    
    real_cell = torch.Tensor([1., 2., 1.])
    cutoff = 1.8
    switching_function=SwitchingFunctions(in_features=n_atoms*3, name='Fermi', cutoff=cutoff, options={'q':0.01})
  
    model = EigsAdjMat(mode = 'continuous',
                       cutoff = cutoff, 
                       n_atoms = n_atoms,
                       PBC = True,
                       real_cell = real_cell,
                       scaled_coords = False,
                       switching_function=switching_function)
    out = model(pos)

    model = EigsAdjMat(mode = 'continuous',
                       cutoff = cutoff, 
                       n_atoms = n_atoms,
                       PBC = True,
                       real_cell = real_cell,
                       scaled_coords = False,
                       switching_function=switching_function)
    out = model(pos)
    assert(out.shape[-1] == model.out_features)

if __name__ == "__main__":
    test_eigs_of_adj_matrix()