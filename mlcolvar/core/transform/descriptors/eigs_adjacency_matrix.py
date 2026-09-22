import torch

from mlcolvar.core.transform import Transform
from mlcolvar.core.transform.descriptors.utils import compute_adjacency_matrix, _resolve_descriptor_cell, sanitize_cell_shape

from typing import Union

__all__ = ["EigsAdjMat"]

class EigsAdjMat(Transform):
    """
    Eigenvalues of the adjacency matrix for a set of atoms from their positions
    """

    def __init__(self,
                 mode: str,
                 cutoff: float, 
                 n_atoms: int,
                 PBC: bool,
                 cell: Union[float, list, None] = None,
                 scaled_coords: bool = False,
                 switching_function = None) -> torch.Tensor:
        """Initialize an eigenvalues of an adjacency matrix object.
           The cell size to be used for PBC and/or scaled coordinates needs to be provided.
           This can be done in one of two ways, exclusively:
           - Fixed cell (e.g., NVT simulations), at initialization, using the `cell` keyword, for a fixed cell only. 
             This mode supports torchscript of the preprocessing module and can be used with the `PLUMED` interface.
           - Varying cells, at runtime (e.g., NPT simulations), using the `cell` entry in the `forward` method or adding the `cell` data in the dataset used for training. 
             This mode **doesn't** support torchscript of the preprocessing module, as it is not supported in the `PLUMED` interface.

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
        cell : Union[float, list, None]
            Dimensions of the real cell for fixed cell mode, orthorombic-like cells only.
            For varying cell mode, this argument must be left as None and the cell must be provided at runtime.
            Note that only fixed cell mode supports torchscript of the preprocessing module.
        scaled_coords : bool
            Switch for coordinates scaled on cell's vectors use, by default False
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
        default_cell = None if cell is None else sanitize_cell_shape(cell)
        self.register_buffer("default_cell", default_cell)
        self.scaled_coords = scaled_coords
        # Register switching_function as submodule if it's a module, so it moves with the model
        if switching_function is not None and isinstance(switching_function, torch.nn.Module):
            self.add_module('switching_function', switching_function)
        else:
            self.switching_function = switching_function

    def compute_adjacency_matrix(self, pos, cell=None):
        cell = _resolve_descriptor_cell(runtime_cell=cell,
                                       default_cell=self.default_cell,
                                       require_cell=self.PBC or self.scaled_coords,
                                    )
        pos = compute_adjacency_matrix(pos=pos,
                                        mode=self.mode,
                                        cutoff=self.cutoff, 
                                        n_atoms=self.n_atoms,
                                        PBC=self.PBC,
                                        cell=cell,
                                        scaled_coords=self.scaled_coords,
                                        switching_function=self.switching_function)
        return pos
    
    def get_eigenvalues(self, x):
        eigs = torch.linalg.eigvalsh(x)
        return eigs

    def forward(self, x: torch.Tensor, cell: Union[float, list, torch.Tensor] = None):
        x = self.compute_adjacency_matrix(x, cell=cell)
        eigs = self.get_eigenvalues(x)
        return eigs