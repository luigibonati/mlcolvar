import torch
import numpy as np 

from mlcolvar.core.transform import Transform
from mlcolvar.core.transform.utils import compute_distances_components_matrices, sanitize_positions_shape

from typing import Union

__all__ = ["TorsionalAngle"]

class TorsionalAngle(Transform):
    """
    Pairwise distances transform, compute all the non duplicated pairwise distances for a set of atoms from their positions
    """
    
    MODES = ["angle", "sin", "cos"]

    def __init__(self, 
                 indeces : Union[list, np.ndarray, torch.Tensor],
                 n_atoms : int,
                 mode: Union[str, list],
                 PBC: bool,
                 real_cell: Union[float, list],
                 scaled_coords : bool) -> torch.Tensor:
        
        # check mode here to get number of out_features
        for i in mode:
            if i not in self.MODES:
                raise ValueError(f'The mode {i} is not available in this class. The available modes are: {", ".join(self.MODES)}.')

        mode_idx = []
        for i,m in enumerate(['angle', 'sin', 'cos']):
            if m in mode:
                mode_idx.append(i)
        self.mode_idx = mode_idx

        # now we can initialize the mother class
        super().__init__(in_features=int(n_atoms*3), out_features=len(mode_idx))

        # initialize class attributes
        self.indeces = indeces
        self.n_atoms = n_atoms
        self.PBC = PBC
        self.real_cell = real_cell
        self.scaled_coords = scaled_coords


    def compute_torsional_angle(self, pos):
        tors_pos, batch_size = sanitize_positions_shape(pos, self.n_atoms)

        # select relevant atoms only
        tors_pos = tors_pos[:, self.indeces, :]

        dist_components, _, _ = compute_distances_components_matrices(pos = tors_pos,
                                                                      n_atoms = 4,
                                                                      PBC = self.PBC,
                                                                      real_cell = self.real_cell,
                                                                      scaled_coords = self.scaled_coords)

        # get AB, BC, CD distances
        AB = dist_components[:, :, 0, 1]
        BC = dist_components[:, :, 1, 2]
        CD = dist_components[:, :, 2, 3]

        # check that they are in the -0.5 : 0.5 range
        AB = self._center_distances(AB)
        BC = self._center_distances(BC)
        CD = self._center_distances(CD)
        # obtain normal direction 
        n1 = torch.cross(AB, BC)
        n2 = torch.cross(BC, CD)
        # obtain versors
        n1_normalized = n1 / torch.norm(n1, dim=1, keepdim=True)
        n2_normalized = n2 / torch.norm(n2, dim=1, keepdim=True)
        UBC= BC / torch.norm(BC,dim=1,keepdim=True)

        sin = torch.einsum('bij,bij->bj', torch.cross(n1_normalized, n2_normalized).unsqueeze(-1), UBC.unsqueeze(-1))
        cos = torch.einsum('bij,bij->bj', n1_normalized.unsqueeze(-1), n2_normalized.unsqueeze(-1))

        angle = torch.atan2(sin, cos)
        
        return torch.hstack([angle, sin, cos])
    
    def _center_distances(self, dist):
        dist[dist >  0.5] = dist[dist >  0.5] - 1
        dist[dist < -0.5] = dist[dist < -0.5] + 1
        return dist

    def forward(self, x):
        out = self.compute_torsional_angle(x)
        return out[:, self.mode_idx]

def test_torsional_angle():
    pos = torch.Tensor([[[ 0.3887, -0.4169, -0.1212],
         [ 0.4264, -0.4374, -0.0983],
         [ 0.4574, -0.4136, -0.0931],
         [ 0.4273, -0.4797, -0.0871],
         [ 0.4684,  0.4965, -0.0692],
         [ 0.4478,  0.4571, -0.0441],
         [-0.4933,  0.4869, -0.1026],
         [-0.4840,  0.4488, -0.1116],
         [-0.4748, -0.4781, -0.1232],
         [-0.4407, -0.4781, -0.1569]],
        [[ 0.3910, -0.4103, -0.1189],
         [ 0.4334, -0.4329, -0.1020],
         [ 0.4682, -0.4145, -0.1013],
         [ 0.4322, -0.4739, -0.0867],
         [ 0.4669, -0.4992, -0.0666],
         [ 0.4448,  0.4670, -0.0375],
         [-0.4975,  0.4844, -0.0981],
         [-0.4849,  0.4466, -0.0991],
         [-0.4818, -0.4870, -0.1291],
         [-0.4490, -0.4933, -0.1668]]])
    pos.requires_grad = True

    real_cell = torch.Tensor([3.0233, 3.0233, 3.0233])
    model = TorsionalAngle([1,3,4,6], 10, ['angle', 'sin', 'cos'], False, real_cell, False)
    angle = model(pos)
    print(angle)
    angle.sum().backward()

    model = TorsionalAngle([1,3,4,6], 10, ['sin'], False, real_cell, False)
    angle = model(pos)
    print(angle)
    angle.sum().backward()


if __name__ == "__main__":
    test_torsional_angle()
    

