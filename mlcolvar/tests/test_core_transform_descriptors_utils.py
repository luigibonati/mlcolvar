import torch

from mlcolvar.core.transform.descriptors.utils import (
    apply_cutoff,
    compute_adjacency_matrix,
    compute_distances_matrix,
)
from mlcolvar.core.transform.tools.switching_functions import SwitchingFunctions


def test_applycutoff():    
    n_atoms=2
    pos = torch.Tensor([ [ [0., 0., 0.],
                           [1., 1., 1.] ],
                         [ [0., 0., 0.],
                           [1., 1., 1.] ] ]
                      )
    cell = torch.Tensor([1., 2, 1.])
    cutoff = 1.8

    # TEST no scaled coords
    out = compute_distances_matrix(pos=pos, n_atoms=n_atoms, PBC=True, cell=cell, scaled_coords=False)
    switching_function=SwitchingFunctions(in_features=n_atoms**2, name='Fermi', cutoff=cutoff, options={'q':0.01})
    apply_cutoff(x=out, cutoff=cutoff, mode='continuous', switching_function=switching_function)
    
    def silly_switch(x):
        return torch.pow(x, 2)
    switching_function = silly_switch
    apply_cutoff(x=out, cutoff=cutoff, mode='continuous', switching_function=switching_function)
    apply_cutoff(x=out, cutoff=cutoff, mode='discontinuous')

    # TEST scaled coords
    pos = torch.einsum('bij,j->bij', pos, 1/cell)
    out = compute_distances_matrix(pos=pos, n_atoms=2, PBC=True, cell=cell, scaled_coords=True)
    switching_function=SwitchingFunctions(in_features=n_atoms**2, name='Fermi', cutoff=cutoff, options={'q':0.01})
    apply_cutoff(x=out, cutoff=cutoff, mode='continuous', switching_function=switching_function)
    apply_cutoff(x=out, cutoff=cutoff, mode='discontinuous')


def test_adjacency_matrix():    
    n_atoms=2
    pos = torch.Tensor([ [ [0., 0., 0.],
                           [1., 1., 1.] ],
                         [ [0., 0., 0.],
                           [1., 1.1, 1.] ] ]
                      )
    
    cell = torch.Tensor([1., 2., 1.])
    cutoff = 1.8
    switching_function=SwitchingFunctions(in_features=n_atoms*3, name='Fermi', cutoff=cutoff, options={'q' : 0.01})
  
    compute_adjacency_matrix(pos=pos, mode='continuous', cutoff=cutoff,  n_atoms=n_atoms, PBC=True, cell=cell, scaled_coords=False, switching_function=switching_function)
