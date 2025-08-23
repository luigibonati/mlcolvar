import torch
import numpy as np 

from mlcolvar.core.transform import Transform
from mlcolvar.core.transform.descriptors.utils import compute_distances_matrix, sanitize_positions_shape

from typing import Union

__all__ = ["TorsionalAngle"]

class TorsionalAngle(Transform):
    """
    Torsional angle defined by a set of 4 atoms from their positions
    """
    
    MODES = ["angle", "sin", "cos"]

    def __init__(self, 
                 indices: Union[list, np.ndarray, torch.Tensor],
                 n_atoms: int,
                 mode: Union[str, list],
                 PBC: bool,
                 cell: Union[float, list],
                 scaled_coords: bool = False) -> torch.Tensor:
        """Initialize a torsional angle object

        Parameters
        ----------
        indices : Union[list, np.ndarray, torch.Tensor]
            Indices of the ordered atoms defining the torsional angle
        n_atoms : int
            Number of atoms in the positions tensor used in the forward.
        mode : Union[str, list]
            Which quantities to return among 'angle', 'sin' and 'cos'
        PBC : bool
            Switch for Periodic Boundary Conditions use
        cell : Union[float, list]
            Dimensions of the real cell, orthorombic-like cells only
        scaled_coords : bool, optional
            Switch for coordinates scaled on cell's vectors use, by default False

        Returns
        -------
        torch.Tensor
            Depending on `mode` selection, the torsional angle in radiants, its sine and its cosine. 
        """

        indices = torch.Tensor(indices).to(torch.long)

        # check indexes are in the correct form
        if indices.numel() != 4:
            print(indices.numel)
            raise ValueError(f"Only the four atom indeces defining this torsional angle must be provided! Found {indices}")
        
        # check mode here to get number of out_features
        for i in mode:
            if i not in self.MODES:
                raise ValueError(f'The mode {i} is not available in this class. The available modes are: {", ".join(self.MODES)}.')

        mode_idx = []

        for n in mode:
            if n not in self.MODES:
                raise(ValueError(f"The given mode : {n} is not available! The available options are {', '.join(self.MODES)}")) 

        for i,m in enumerate(self.MODES):
            if m in mode:
                mode_idx.append(i)
        self.mode_idx = mode_idx

        # now we can initialize the mother class
        super().__init__(in_features=int(n_atoms*3), out_features=len(mode_idx))

        # initialize class attributes
        self.indices = indices
        self.n_atoms = n_atoms
        self.PBC = PBC
        self.cell = cell
        self.scaled_coords = scaled_coords


    def compute_torsional_angle(self, pos):
        tors_pos, batch_size = sanitize_positions_shape(pos, self.n_atoms)

        # select relevant atoms only
        tors_pos = tors_pos[:, self.indices, :]

        dist_components = compute_distances_matrix(pos=tors_pos,
                                                    n_atoms=4,
                                                    PBC=self.PBC,
                                                    cell=self.cell,
                                                    scaled_coords=self.scaled_coords,
                                                    vector=True)

        # get AB, BC, CD distances
        AB = dist_components[:, :, 0, 1]
        BC = dist_components[:, :, 1, 2]
        CD = dist_components[:, :, 2, 3]

        # obtain normal direction 
        n1 = torch.linalg.cross(AB, BC)
        n2 = torch.linalg.cross(BC, CD)
        # obtain versors
        n1_normalized = n1 / torch.norm(n1, dim=1, keepdim=True)
        n2_normalized = n2 / torch.norm(n2, dim=1, keepdim=True)
        UBC= BC / torch.norm(BC,dim=1,keepdim=True)

        sin = torch.einsum('bij,bij->bj', torch.linalg.cross(n1_normalized, n2_normalized).unsqueeze(-1), UBC.unsqueeze(-1))
        cos = torch.einsum('bij,bij->bj', n1_normalized.unsqueeze(-1), n2_normalized.unsqueeze(-1))

        angle = torch.atan2(sin, cos)
        
        return torch.hstack([angle, sin, cos])

    def forward(self, x):
        out = self.compute_torsional_angle(x)
        return out[:, self.mode_idx]

def test_torsional_angle():
    # simple test on alanine phi angle
    pos = torch.Tensor([[ 1.4970,  1.3861, -0.0273, -1.4933,  1.5070, -0.1133, -1.4473, -1.4193,
                        -0.0553,  1.4940,  1.4990, -0.2403,  1.4780, -1.4173, -0.3363, -1.4243,
                        -1.4093, -0.4293,  1.3530, -1.4313, -0.4183,  1.3060,  1.4750, -0.4333,
                        1.2970, -1.3233, -0.4643,  1.1670, -1.3253, -0.5354],
                        [ 1.4932,  1.3759, -0.0133,  1.4651,  1.4984, -0.1044,  1.4294, -1.4193,
                        -0.0564,  1.4877,  1.4869, -0.2352,  1.4949, -1.4240, -0.3326, -1.3827,
                        -1.4189, -0.3863,  1.3877, -1.4264, -0.4353,  1.3727,  1.4973, -0.5063,
                        1.3086, -1.3215, -0.4382,  1.2070, -1.3063, -0.5443]])
    pos.requires_grad = True

    ref_phi = torch.Tensor([[-2.3687], [-2.0190]])
    cell = torch.Tensor([3.0233, 3.0233, 3.0233])

    model = TorsionalAngle(indices=[1,3,4,6], n_atoms=10, mode=['angle'], PBC=True, cell=cell, scaled_coords=False)
    angle = model(pos)
    assert(torch.allclose(angle, ref_phi, atol=1e-3))
    angle.sum().backward()

    model = TorsionalAngle(np.array([1,3,4,6]), n_atoms=10, mode=['angle', 'sin', 'cos'], PBC=True, cell=cell, scaled_coords=False)
    angle = model(pos)
    angle.sum().backward()
    assert(torch.allclose(angle[:, 0].unsqueeze(-1), ref_phi, atol=1e-3))
    assert(torch.allclose(angle[:, 1].unsqueeze(-1), torch.sin(ref_phi), atol=1e-3))
    assert(torch.allclose(angle[:, 2].unsqueeze(-1), torch.cos(ref_phi), atol=1e-3))


    model = TorsionalAngle(torch.Tensor([1,3,4,6]), n_atoms=10, mode=['sin', 'cos'], PBC=True, cell=cell, scaled_coords=False)
    angle = model(pos)
    angle.sum().backward()
    assert(torch.allclose(angle[:, 0].unsqueeze(-1), torch.sin(ref_phi), atol=1e-3))
    assert(torch.allclose(angle[:, 1].unsqueeze(-1), torch.cos(ref_phi), atol=1e-3))

    # TODO add reference value for check

if __name__ == "__main__":
    test_torsional_angle()
    

