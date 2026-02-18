import torch
import numpy as np 

from mlcolvar.core.transform import Transform
from mlcolvar.core.transform.descriptors.utils import compute_distances_matrix, sanitize_positions_shape

from typing import Union

__all__ = ["TorsionalAngles"]

def TorsionalAngle(*args, **kwargs):
    raise DeprecationWarning("The class name TorsionalAngles has been deprecated use TorsionalAngless instead!")

class TorsionalAngles(Transform):
    """
    Torsional angle defined by a set of 4 atoms from their positions.
    Can compute a single angle or multiple angles.
    """
    
    MODES = ["angle", "sin", "cos"]

    def __init__(self, 
                 indices: Union[list, np.ndarray, torch.Tensor],
                 n_atoms: int,
                 mode: Union[str, list],
                 PBC: bool,
                 cell: Union[float, list],
                 scaled_coords: bool = False) -> torch.Tensor:
        """Initialize a torsional angle object.
           Can compute a single angle or multiple angles based on the `indices` key.

        Parameters
        ----------
        indices : Union[list, np.ndarray, torch.Tensor]
            Indices of the 4 ordered atoms defining the torsional angle(s). 
            It can be:
            - A single 4-element list/array: [a1, a2, a3, a4] for one angle
            - A list of 4-element lists: [[a1, a2, a3, a4], [b1, b2, b3, b4], ...] for multiple angles
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
            Depending on `mode` selection, the torsional angle(s) in radiants, their sine and their cosine. 
            Shape: [batch_size, n_angles * n_modes]
        """

        # Convert indices to tensor and check format
        indices = torch.as_tensor(indices, dtype=torch.long)
        
        # Check if we have a single angle or multiple angles
        if indices.ndim == 1:
            # Single angle: shape should be [4]
            if indices.numel() != 4:
                raise ValueError(f"Single angle must have exactly 4 atom indices! Found {indices.numel()}")
            self.multiple_angles = False
            self.n_angles = 1
            # Reshape to [1, 4] for consistent processing
            indices = indices.unsqueeze(0)
        elif indices.ndim == 2:
            # Multiple angles: shape should be [n_angles, 4]
            if indices.shape[1] != 4:
                raise ValueError(f"Each angle must have exactly 4 atom indices! Found shape {indices.shape}")
            self.multiple_angles = True
            self.n_angles = indices.shape[0]
        else:
            raise ValueError(f"Indices must be 1D (single angle) or 2D (multiple angles)! Found shape {indices.shape}")
        
        # check mode here to get number of out_features
        if isinstance(mode, str):
            mode = [mode]
        
        for i in mode:
            if i not in self.MODES:
                raise ValueError(f'The mode {i} is not available in this class. The available modes are: {", ".join(self.MODES)}.')

        mode_idx = []
        for i,m in enumerate(self.MODES):
            if m in mode:
                mode_idx.append(i)
        self.mode_idx = mode_idx

        # now we can initialize the mother class
        # out_features = n_angles * n_modes
        super().__init__(in_features=int(n_atoms*3), out_features=self.n_angles * len(mode_idx))

        # initialize class attributes
        self.register_buffer('indices', indices)
        self.register_buffer('cell', torch.as_tensor(cell))
        self.n_atoms = n_atoms
        self.PBC = PBC
        self.scaled_coords = scaled_coords


    def compute_torsional_angle(self, pos):
        tors_pos, batch_size = sanitize_positions_shape(pos, self.n_atoms)

        # indices: [n_angles, 4]
        angle_indices = self.indices  # assumed LongTensor

        # Gather all torsion atom positions at once
        # Result: [batch_size, n_angles, 4, 3]
        angle_pos = tors_pos[:, angle_indices, :]

        # Merge batch and angle dims to reuse your distance function
        B, A = batch_size, self.n_angles
        angle_pos_reshaped = angle_pos.reshape(B * A, 4, 3)

        dist_components = compute_distances_matrix(
            pos=angle_pos_reshaped,
            n_atoms=4,
            PBC=self.PBC,
            cell=self.cell,
            scaled_coords=self.scaled_coords,
            vector=True
        )

        # dist_components: [B*A, 3, 4, 4]
        AB = dist_components[:, :, 0, 1]
        BC = dist_components[:, :, 1, 2]
        CD = dist_components[:, :, 2, 3]
        # Cross products
        n1 = torch.linalg.cross(AB, BC, dim=1)
        n2 = torch.linalg.cross(BC, CD, dim=1)

        # Normalize
        n1_normalized = n1 / torch.norm(n1, dim=1, keepdim=True)
        n2_normalized = n2 / torch.norm(n2, dim=1, keepdim=True)
        UBC = BC / torch.norm(BC, dim=1, keepdim=True)


        # Compute sin and cos
        sin = torch.sum(torch.linalg.cross(n1_normalized, n2_normalized, dim=1) * UBC, dim=1)
        cos = torch.sum(n1_normalized * n2_normalized, dim=1)


        angle = torch.atan2(sin, cos)

        # Reshape back to [batch_size, n_angles]
        angle = angle.view(B, A, 1)
        sin = sin.view(B, A, 1)
        cos = cos.view(B, A, 1)

        # Stack as [batch_size, n_angles * 3]
        return torch.cat([angle, sin, cos], dim=2)

    def forward(self, x):
        # Ensure x is on the same device as model buffers
        if isinstance(x, torch.Tensor) and x.device != self.indices.device:
            x = x.to(self.indices.device)
        out = self.compute_torsional_angle(x)
        # Select the requested modes for all angles
        # out shape: [batch_size, n_angles * 3]
        # We need to reshape, select modes, and reshape back
        batch_size = out.shape[0]
        out_reshaped = out.view(batch_size, self.n_angles, 3)  # [batch_size, n_angles, 3]
        out_selected = out_reshaped[:, :, self.mode_idx]  # [batch_size, n_angles, n_modes]
        return out_selected.view(batch_size, -1)  # [batch_size, n_angles * n_modes]

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

    # Test single angle (backward compatible)
    model = TorsionalAngles(indices=[1,3,4,6], n_atoms=10, mode=['angle'], PBC=True, cell=cell, scaled_coords=False)
    angle = model(pos)
    assert(torch.allclose(angle, ref_phi, atol=1e-3))
    angle.sum().backward()

    # Test multiple angles
    model = TorsionalAngles(indices=[[1,3,4,6], [1,3,4,6]], n_atoms=10, mode=['angle'], PBC=True, cell=cell, scaled_coords=False)
    angle = model(pos)
    assert(angle.shape == (2, 2))  # [batch_size, n_angles]
    assert(torch.allclose(angle[:, 0], ref_phi.squeeze(), atol=1e-3))
    assert(torch.allclose(angle[:, 1], ref_phi.squeeze(), atol=1e-3))
    angle.sum().backward()

    # Test multiple angles with multiple modes
    model = TorsionalAngles(indices=[[1,3,4,6], [1,3,4,6]], n_atoms=10, mode=['angle', 'sin', 'cos'], PBC=True, cell=cell, scaled_coords=False)
    angle = model(pos)
    assert(angle.shape == (2, 6))  # [batch_size, n_angles * n_modes]
    angle.sum().backward()

    # Original tests
    model = TorsionalAngles(np.array([1,3,4,6]), n_atoms=10, mode=['angle', 'sin', 'cos'], PBC=True, cell=cell, scaled_coords=False)
    angle = model(pos)
    angle.sum().backward()
    assert(torch.allclose(angle[:, 0].unsqueeze(-1), ref_phi, atol=1e-3))
    assert(torch.allclose(angle[:, 1].unsqueeze(-1), torch.sin(ref_phi), atol=1e-3))
    assert(torch.allclose(angle[:, 2].unsqueeze(-1), torch.cos(ref_phi), atol=1e-3))

    model = TorsionalAngles(torch.Tensor([1,3,4,6]), n_atoms=10, mode=['sin', 'cos'], PBC=True, cell=cell, scaled_coords=False)
    angle = model(pos)
    angle.sum().backward()
    assert(torch.allclose(angle[:, 0].unsqueeze(-1), torch.sin(ref_phi), atol=1e-3))
    assert(torch.allclose(angle[:, 1].unsqueeze(-1), torch.cos(ref_phi), atol=1e-3))

if __name__ == "__main__":
    test_torsional_angle()