import torch

from mlcolvar.core.transform import Transform
from mlcolvar.core.transform.descriptors.utils import compute_distances_matrix, compute_distances_pairs, _resolve_descriptor_cell, sanitize_cell_shape

from typing import Union

__all__ = ["PairwiseDistances"]

class PairwiseDistances(Transform):
    """
    Non duplicated pairwise distances for a set of atoms from their positions.
    Can compute either all the distances or a subset.
    """

    def __init__(self, 
                 n_atoms: int,
                 PBC: bool,
                 cell: Union[float, list, None] = None,
                 scaled_coords: bool = False,
                 slicing_pairs: list = None) -> torch.Tensor:
        """Initialize a pairwise distances object.
        Can compute either all the distances or a subset based on the `slicing_pairs` key.
        The cell size to be used for PBC and/or scaled coordinates needs to be provided.
        This can be done in one of two ways, exclusively:
        - Fixed cell (e.g., NVT simulations), at initialization, using the `cell` keyword, for a fixed cell only. 
          This mode supports torchscript of the preprocessing module and can be used with the `PLUMED` interface.
        - Varying cells, at runtime (e.g., NPT simulations), using the `cell` entry in the `forward` method or adding the `cell` data in the dataset used for training. 
          This mode **doesn't** support torchscript of the preprocessing module, as it is not supported in the `PLUMED` interface.

        Parameters
        ----------
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
        slicing_pairs : list
            indices of the subset of distances to be returned, by default None

        Returns
        -------
        torch.Tensor
            Non duplicated pairwise distances
        """
        if slicing_pairs is None:
            super().__init__(in_features=int(n_atoms*3), out_features=int(n_atoms*(n_atoms-1) / 2))
        else: 
            super().__init__(in_features=int(n_atoms*3), out_features=len(slicing_pairs))

        # parse args
        self.n_atoms = n_atoms
        self.PBC = PBC
        self.scaled_coords = scaled_coords
        default_cell = None if cell is None else sanitize_cell_shape(cell)
        self.register_buffer("default_cell", default_cell)
        self.register_buffer('slicing_pairs', 
                                torch.tensor(slicing_pairs, dtype=torch.long) if slicing_pairs is not None else None)

    def compute_pairwise_distances(self, pos, cell=None):
        cell = _resolve_descriptor_cell(runtime_cell=cell,
                                       default_cell=self.default_cell,
                                       require_cell=self.PBC or self.scaled_coords,
                                    )
        # if we compute all distances we use the matrix trick
        if self.slicing_pairs is None:
            dist = compute_distances_matrix(pos=pos,
                                            n_atoms=self.n_atoms,
                                            PBC=self.PBC,
                                            cell=cell,
                                            scaled_coords=self.scaled_coords)
            batch_size = dist.shape[0]
            device = pos.device
            # mask out diagonal elements
            aux_mask = torch.ones_like(dist, device=device) - torch.eye(dist.shape[-1], device=device)
            # keep upper triangular part to avoid duplicates
            unique = aux_mask.triu().nonzero(as_tuple=True)
            pairwise_distances = dist[unique].reshape((batch_size, -1)) 
        
        # if we only compute a few selected distances we do that explicitly
        else:
            dist = compute_distances_pairs(pos=pos,
                                           n_atoms=self.n_atoms,
                                           PBC=self.PBC,
                                           cell=cell,
                                           scaled_coords=self.scaled_coords,
                                           slicing_pairs=self.slicing_pairs)
        
            pairwise_distances = dist
        return pairwise_distances
        

    def forward(self, x: torch.Tensor, cell: Union[float, list, torch.Tensor] = None):
        x = self.compute_pairwise_distances(x, cell=cell)
        return x