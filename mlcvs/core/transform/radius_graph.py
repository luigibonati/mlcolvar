import torch

from mlcvs.core.transform import Transform
from mlcvs.core.transform.utils import compute_distances_matrix,apply_cutoff

from typing import Union

__all__ = ["RadiusGraph"]

class RadiusGraph(Transform):
    """
    Radius Graph transform, compute the elements to build a distance-cutoff based graph for a set of atoms from their positions
    """

    def __init__(self,
                 mode : str,
                 cutoff : float, 
                 n_atoms : int,
                 PBC: bool,
                 real_cell: Union[float, list],
                 scaled_coords : bool,
                 switching_function = None,
                 zero_threshold : float = 1e-4) -> torch.Tensor:
        """Initialize a radius graph object

        Parameters
        ----------
        mode : str
            Mode for cutoff application, either:
            - 'continuous': applies a switching function to the distances which can be specified with switching_function keyword and keep only what is >= zero_threshold, has stable derivatives
            - 'discontinuous': keep what is >= cutoff and discard the rest, derivatives may be be incorrect
        cutoff : float
            Radial cutoff for the connectivity criterion
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
        zero_threshold : float, optional
            Threshold to be considered zero when in continuous mode, it is ignored if in discontinuous mode, by default 1e-4

        Returns
        -------
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
            Elements to build a distance-cutoff based graph:
            - Distances for each edge
            - Batch indeces
            - Edge source indeces
            - Edge destination indeces


        """
        super().__init__(in_features=int(n_atoms*3), out_features=None)

        # parse args
        self.mode = mode
        self.cutoff = cutoff 
        self.n_atoms = n_atoms
        self.PBC = PBC
        self.real_cell = real_cell
        self.scaled_coords = scaled_coords
        self.switching_function = switching_function
        self.zero_threshold = zero_threshold
        
    def compute_radius_graph(self, pos, mode):
        dist = compute_distances_matrix(pos=pos,
                                        n_atoms=self.n_atoms,
                                        PBC=self.PBC,
                                        real_cell=self.real_cell,
                                        scaled_coords=self.scaled_coords)
        
        # we want to smooth dist at cutoff, we first get a switch and then apply that to dist
        # we need a clone! Otherwise dist would be modified!
        aux_switch = torch.clone(dist)
        aux_switch = apply_cutoff(x=aux_switch,
                                  cutoff=self.cutoff, 
                                  mode=mode, 
                                  switching_function = self.switching_function)          

        if mode == 'continuous':
            # smooth dist
            dist = dist * aux_switch 
            # discard what is almost zero --> use self.zero_threshold
            unique = torch.nonzero(torch.ge(dist.triu(), self.zero_threshold), as_tuple=True)

        elif mode == 'discontinuous': 
            # discard zeros entries
            unique = dist.triu().nonzero(as_tuple=True) 
        
        distances = dist[unique]
        batch_indeces, edge_src, edge_dst = unique
        
        return distances,batch_indeces,edge_src,edge_dst

    def forward(self, x : torch.Tensor):
        x = self.compute_radius_graph(x, mode=self.mode)
        return x

def test_radiusgraph():
    from mlcvs.core.transform.switching_functions import SwitchingFunctions

    n_atoms=3
    pos = torch.Tensor([ [ [0., 0., 0.],
                           [1., 1., 1.] ],
                         [ [0., 0., 0.],
                           [1., 1.1, 1.] ] ]
                      )
    
    real_cell = torch.Tensor([1., 2., 1.])
    cutoff = 1.8
    switching_function=SwitchingFunctions(in_features=n_atoms*3, name='Fermi', cutoff=cutoff, options={'q':0.01})
    model = RadiusGraph(mode = 'continuous',
                        cutoff = cutoff, 
                        n_atoms = 2,
                        PBC = True,
                        real_cell = real_cell,
                        scaled_coords = False,
                        switching_function=switching_function)
    distances,batch_indeces,edge_src,edge_dst = model(pos)

if __name__ == "__main__":
    test_radiusgraph()
              