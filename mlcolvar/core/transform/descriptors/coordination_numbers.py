import torch
import numpy as np

from mlcolvar.core.transform import Transform
from mlcolvar.core.transform.descriptors.utils import apply_cutoff, compute_distances_matrix, _resolve_descriptor_cell, sanitize_cell_shape, sanitize_positions_shape
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
                 cell: Union[float, list, None] = None,
                 mode: str = 'continuous',
                 scaled_coords: bool = False,
                 switching_function = None, 
                 dmax: float = None) -> torch.Tensor:
        """Initialize a coordination number object between two groups of atoms A and B.
           The cell size to be used for PBC and/or scaled coordinates needs to be provided.
           This can be done in one of two ways, exclusively:
           - Fixed cell (e.g., NVT simulations), at initialization, using the `cell` keyword, for a fixed cell only. 
             This mode supports torchscript of the preprocessing module and can be used with the `PLUMED` interface.
           - Varying cells, at runtime (e.g., NPT simulations), using the `cell` entry in the `forward` method or adding the `cell` data in the dataset used for training. 
             This mode **doesn't** support torchscript of the preprocessing module, as it is not supported in the `PLUMED` interface.
        
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
        cell : Union[float, list, None]
            Dimensions of the real cell for fixed cell mode, orthorombic-like cells only.
            For varying cell mode, this argument must be left as None and the cell must be provided at runtime.
            Note that only fixed cell mode supports torchscript of the preprocessing module.
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

        # parse args
        self.group_A = group_A
        self._group_A_size = len(group_A)
        self.group_B = group_B
        self._group_B_size = len(group_B)
        self._n_used_atoms = self._group_A_size + self._group_B_size
        self.cutoff = cutoff
        self.n_atoms = n_atoms
        self.PBC = PBC
        self.scaled_coords = scaled_coords
        self.mode = mode
        self.dmax = dmax

        
        # register buffers that should move with the model
        reordering = np.concatenate((self.group_A, self.group_B))
        self.register_buffer('_reordering', torch.tensor(reordering, dtype=torch.long))
        default_cell = None if cell is None else sanitize_cell_shape(cell)
        self.register_buffer("default_cell", default_cell)
        
        # register switching_function as submodule if it's a Module
        if switching_function is not None and isinstance(switching_function, torch.nn.Module):
            self.add_module('switching_function', switching_function)
        else:
            self.switching_function = switching_function

        
    def compute_coordination_number(self, pos, cell=None):
        cell = _resolve_descriptor_cell(runtime_cell=cell,
                                       default_cell=self.default_cell,
                                       require_cell=self.PBC or self.scaled_coords,
                                    )
        # move the group A elements to first positions
        pos, batch_size = sanitize_positions_shape(pos, self.n_atoms)
        pos = pos[:, self._reordering, :]
        dist = compute_distances_matrix(pos=pos,
                                        n_atoms=self._n_used_atoms,
                                        PBC=self.PBC,
                                        cell=cell,
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
    
    def forward(self, pos, cell: Union[float, list, torch.Tensor] = None):
        coord_numbers = self.compute_coordination_number(pos, cell=cell)
        return coord_numbers    
