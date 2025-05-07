import torch
import numpy as np

from mlcolvar.core.transform import Transform
from mlcolvar.core.transform.descriptors.utils import compute_distances_matrix, apply_cutoff, sanitize_positions_shape

from typing import Union

from warnings import warn

__all__ = ["CoordinationNumbers"]

class CoordinationNumbers(Transform):
    """
    Coordination number between the elements of two groups of atoms from their positions
    """

    def __init__(self,
                 group_A: list,
                 group_B: list,
                 cutoff: float,
                 n_atoms: int,
                 PBC: bool,
                 cell: Union[float, list],
                 mode: str,
                 scaled_coords: bool = False,
                 switching_function = None, 
                 dmax: float = None) -> torch.Tensor:
        """Initialize a coordination number object between two groups of atoms A and B.

        Parameters
        ----------
        group_A : list
            Zero-based indices of group A atoms
        group_B : list
            Zero-based indices of group B atoms
        cutoff : float
            Cutoff radius for coordination number evaluation
        n_atoms : int
            Total number of atoms in the system
        PBC : bool
            Switch for Periodic Boundary Conditions use
        cell : Union[float, list]
            Dimensions of the real cell, orthorombic-like cells only
        mode : str
            Mode for cutoff application, either:
            - 'continuous': applies a switching function to the distances which can be specified with switching_function keyword, has stable derivatives
            - 'discontinuous': set at zero everything above the cutoff and one below, derivatives may be be incorrect        
        scaled_coords : bool
            Switch for coordinates scaled on cell's vectors use, by default False
        switching_function : _type_, optional
            Switching function to be applied for the cutoff, can be either initialized as a switching_functions/SwitchingFunctions class or a simple function, by default None
        dmax : float, optional
            Distance at which, if set, the switching function will be forced to be zero by strecthing it and shifting it, by default None.

        Returns
        -------
        torch.Tensor
            Coordination numbers of elements of group A with respect to elements of group B
        """
        super().__init__(in_features=int(n_atoms*3), out_features=len(group_A))
        
        # parse args
        self.group_A = group_A
        self._group_A_size = len(group_A)
        self.group_B = group_B
        self._group_B_size = len(group_B)
        self._n_used_atoms = self._group_A_size + self._group_B_size
        self._reordering = np.concatenate((self.group_A, self.group_B))
        self.cutoff = cutoff
        self.n_atoms = n_atoms
        self.PBC = PBC
        self.cell = cell
        self.scaled_coords = scaled_coords
        self.mode = mode
        self.switching_function = switching_function
        self.dmax = dmax

        # do a few checks
        if mode == 'continuous':
            if switching_function is None:
                raise ValueError('switching_function is required to use continuous mode! Set This can be either a user-defined and torch-based function or a method of class switching_functions/SwitchingFunctions')
            if cutoff != switching_function.cutoff:
                raise ValueError(f'The cutoff of CoordinationNumbers and switching_function must be the same! Found {cutoff} and {switching_function.cutoff}')
            if dmax is not None and dmax != switching_function.dmax:
                raise ValueError(f'The dmax of CoordinationNumbers and switching_function must be the same! Found {dmax} and {switching_function.dmax}')
        if mode == 'discontinuous':
            if dmax is not None:
                warn('dmax was set in discontinuous mode, it will likely be ineffective!')
        
    def compute_coordination_number(self, pos):
        # move the group A elements to first positions
        pos, batch_size = sanitize_positions_shape(pos, self.n_atoms)
        pos = pos[:, self._reordering, :]
        dist = compute_distances_matrix(pos=pos,
                                        n_atoms=self._n_used_atoms,
                                        PBC=self.PBC,
                                        cell=self.cell,
                                        scaled_coords=self.scaled_coords)

        # get mask in case dmax is set
        mask_dmax = torch.ones_like(dist)
        if self.dmax is not None:
            mask_dmax[torch.nonzero(dist.gt(self.dmax), as_tuple=True)] = 0

        # we can apply the switching cutoff with the switching function
        contributions = apply_cutoff(x=dist, 
                            cutoff=self.cutoff, 
                            mode=self.mode, 
                            switching_function=self.switching_function)
        
        # we can throw away part of the matrix as it is repeated uselessly
        contributions = contributions[:, :self._group_A_size, :]
        mask_dmax = mask_dmax[:, :self._group_A_size, :]

        # and also ensure that the AxA part of the matrix is zero, we need also to preserve the gradients
        mask = torch.ones_like(contributions)
        mask[:, :self._group_A_size, :self._group_A_size] = 0
        contributions = contributions*mask
        contributions = contributions*mask_dmax

        # compute coordination
        coord_numbers = torch.sum(contributions, dim=-1)

        return coord_numbers
    
    def forward(self, pos):
        coord_numbers = self.compute_coordination_number(pos)
        return coord_numbers    
    

def test_coordination_number():
    from mlcolvar.core.transform.tools.switching_functions import SwitchingFunctions
    
    # simple example based on calixarene water coordination numbers
    pos = torch.Tensor([[[-0.410219, -0.680065, -2.016121],
                         [-0.164329, -0.630426, -2.120843],
                         [-0.250341, -0.392700, -1.534535],
                         [-0.277187, -0.615506, -1.335904],
                         [-0.762276, -1.041939, -1.546581],
                         [-0.200766, -0.851481, -1.534129],
                         [ 0.051099, -0.898884, -1.628219],
                         [-1.257225,  1.671602,  0.166190],
                         [-0.486917, -0.902610, -1.554715],
                         [-0.020386, -0.566621, -1.597171],
                         [-0.507683, -0.541252, -1.540805],
                         [-0.527323, -0.206236, -1.532587]],
                        [[-0.410387, -0.677657, -2.018355],
                         [-0.163502, -0.626094, -2.123348],
                         [-0.250672, -0.389610, -1.536810],
                         [-0.275395, -0.612535, -1.338175],
                         [-0.762197, -1.037856, -1.547382],
                         [-0.200948, -0.847825, -1.536010],
                         [ 0.051170, -0.896311, -1.629396],
                         [-1.257530,  1.674078, 0.165089],
                         [-0.486894, -0.900076, -1.556366],
                         [-0.020235, -0.563252, -1.601229],
                         [-0.507242, -0.537527, -1.543025],
                         [-0.528576, -0.202031, -1.534733]]])

    cell = 4.0273098
    pos.requires_grad = True

    n_atoms = 12
    cutoff=0.25
    switching_function=SwitchingFunctions(in_features=n_atoms*3, name='Rational', cutoff=cutoff, options={'n': 2, 'm' : 6, 'eps' : 1e0})

    model = CoordinationNumbers(group_A=[0, 1],
                                group_B=[2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                                cutoff=cutoff,
                                n_atoms=n_atoms, 
                                PBC=True,
                                cell=cell,
                                mode='continuous',
                                scaled_coords=False,
                                switching_function=switching_function)
    
    out = model(pos)
    out.sum().backward()

    # we swap by hand the 0,1 atoms with 2,3
    pos = torch.Tensor([[[-0.250341, -0.392700, -1.534535],
                         [-0.277187, -0.615506, -1.335904],
                         [-0.410219, -0.680065, -2.016121],
                         [-0.164329, -0.630426, -2.120843],
                         [-0.762276, -1.041939, -1.546581],
                         [-0.200766, -0.851481, -1.534129],
                         [ 0.051099, -0.898884, -1.628219],
                         [-1.257225,  1.671602,  0.166190],
                         [-0.486917, -0.902610, -1.554715],
                         [-0.020386, -0.566621, -1.597171],
                         [-0.507683, -0.541252, -1.540805],
                         [-0.527323, -0.206236, -1.532587]],
                        [[-0.250672, -0.389610, -1.536810],
                         [-0.275395, -0.612535, -1.338175],
                         [-0.410387, -0.677657, -2.018355],
                         [-0.163502, -0.626094, -2.123348],
                         [-0.762197, -1.037856, -1.547382],
                         [-0.200948, -0.847825, -1.536010],
                         [ 0.051170, -0.896311, -1.629396],
                         [-1.257530,  1.674078, 0.165089],
                         [-0.486894, -0.900076, -1.556366],
                         [-0.020235, -0.563252, -1.601229],
                         [-0.507242, -0.537527, -1.543025],
                         [-0.528576, -0.202031, -1.534733]]])
    
    pos.requires_grad = True
    switching_function=SwitchingFunctions(in_features=n_atoms*3, name='Rational', cutoff=cutoff, options={'n': 2, 'm' : 6, 'eps' : 1e0})

    model = CoordinationNumbers(group_A=[2, 3],
                                group_B=[0, 1, 4, 5, 6, 7, 8, 9, 10, 11],
                                cutoff=cutoff,
                                n_atoms=n_atoms, 
                                PBC=True,
                                cell=cell,
                                mode='continuous',
                                scaled_coords=False,
                                switching_function=switching_function)
    
    out_2 = model(pos)
    out_2.sum().backward()
    assert(torch.allclose(out, out_2))

    # check using only subset of atoms
    model = CoordinationNumbers(group_A=[2, 3],
                                group_B=[0, 1, 4, 5, 6],
                                cutoff=cutoff,
                                n_atoms=n_atoms, 
                                PBC=True,
                                cell=cell,
                                mode='continuous',
                                scaled_coords=False,
                                switching_function=switching_function)
    
    out = model(pos)
    out.sum().backward()
    
    # check using dmax
    switching_function=SwitchingFunctions(in_features=n_atoms*3, name='Rational', cutoff=cutoff, dmax=0.6, options={'n': 2, 'm' : 6, 'eps' : 1e0})
    model = CoordinationNumbers(group_A=[2, 3],
                                group_B=[0, 1, 4, 5, 6, 7, 8, 9, 10, 11],
                                cutoff=cutoff,
                                n_atoms=n_atoms, 
                                PBC=True,
                                cell=cell,
                                mode='continuous',
                                scaled_coords=False,
                                switching_function=switching_function,
                                dmax=0.6)
    
    out = model(pos)
    out.sum().backward()


    # TODO add reference value for check

if __name__ == "__main__":
    test_coordination_number()
        
