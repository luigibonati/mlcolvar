import torch
import numpy as np

from mlcolvar.core.transform import Transform
from mlcolvar.core.transform.utils import compute_distances_matrix, apply_cutoff

from typing import Union

__all__ = ["CoordinationNumbers"]

class CoordinationNumbers(Transform):
    """
    Coordination number transform, compute the coordination number between the elements of two groups of atoms for a set of atoms from their positions
    """

    def __init__(self,
                 group_A : list,
                 group_B : list,
                 cutoff : float,
                 n_atoms : int,
                 PBC : bool,
                 real_cell : Union[float, list],
                 scaled_coords : bool,
                 mode : str, 
                 switching_function = None) -> torch.Tensor:
        """Initialize a coordination number object between two groups of atoms A and B.

        Parameters
        ----------
        group_A : list
            Zero-based indeces of group A atoms
        group_B : list
            Zero-based indeces of group B atoms
        cutoff : float
            Cutoff radius for coordination number evaluation
        n_atoms : int
            Total number of atoms in the system
        PBC : bool
            Switch for Periodic Boundary Conditions use
        real_cell : Union[float, list]
            Dimensions of the real cell, orthorombic-like cells only
        scaled_coords : bool
            Switch for coordinates scaled on cell's vectors use
        mode : str
            Mode for cutoff application, either:
            - 'continuous': applies a switching function to the distances which can be specified with switching_function keyword, has stable derivatives
            - 'discontinuous': set at zero everything above the cutoff and one below, derivatives may be be incorrect        
        switching_function : _type_, optional
            Switching function to be applied for the cutoff, can be either initialized as a switching_functions/SwitchingFunctions class or a simple function, by default None

        Returns
        -------
        torch.Tensor
            Coordination numbers of elements of group A with respect to elements of group B
        """
        super().__init__(in_features=int(n_atoms*3), out_features=len(group_A))
        
        # parse args
        self.group_A = group_A
        self.group_A_size = len(group_A)
        self.group_B = group_B
        self.group_B_size = len(group_B)
        self.reordering = np.concatenate([self.group_A, self.group_B])

        self.cutoff = cutoff

        self.n_atoms = n_atoms
        self.PBC = PBC
        self.real_cell = real_cell
        self.scaled_coords = scaled_coords

        self.mode = mode
        self.switching_function = switching_function
    
    def compute_coordination_number(self, pos):
        # move the group A elements to first positions
        # pos = pos[self.reordering]
        dist = compute_distances_matrix(pos=pos,
                                        n_atoms=self.n_atoms,
                                        PBC=self.PBC,
                                        real_cell=self.real_cell,
                                        scaled_coords=self.scaled_coords)
        # batch_size = dist.shape[0]
        # device = pos.device

        # we can apply the switching cutoff with the switching function
        contributions = apply_cutoff(x=dist, 
                            cutoff=self.cutoff, 
                            mode=self.mode, 
                            switching_function=self.switching_function)
        
        # we can throw away part of the matrix as it is repeated uselessly
        contributions = contributions[:, :self.group_A_size, :]

        # and also ensure that the AxA part of the matrix is zero, we need also to preserve the gradients
        mask = torch.ones_like(contributions)
        mask[:, :self.group_A_size, :self.group_A_size] = 0
        contributions = contributions*mask

        # compute coordination
        coord_numbers = torch.sum(contributions, dim=-1)

        return coord_numbers
    
    def forward(self, pos):
        coord_numbers = self.compute_coordination_number(pos)
        return coord_numbers    
    

# def test_coordination_number():
#     from mlcolvar.core.transform.switching_functions import SwitchingFunctions

#     pos = torch.Tensor([[3.057573, -0.970874, 0.010963],
#                         [3.303463, -0.921235, -0.093758],
#                         [3.143809, -0.969587, 0.243834],
#                         [3.144826, -1.055678, 1.234909],
#                         [2.817620, -0.882992, 0.701040],
#                         [3.523643, -0.600038, 1.250236],
#                         [3.047862, -0.612398, 0.731384],
#                         [3.029544, -0.438634, 1.131293],
#                         [3.058277, -1.465733, 1.020088],
#                         [3.267026, -1.142289, 0.492955]])
#     pos.requires_grad = True
    
#     n_atoms = 10
#     cutoff=0.5

#     switching_function=SwitchingFunctions(in_features=n_atoms*3, name='Fermi', cutoff=cutoff, options={'q':0.05})

#     model = CoordinationNumbers(group_A = [0, 1, 2],
#                                 group_B = [3, 4, 5, 6, 7, 8, 9],
#                                 cutoff= cutoff,
#                                 n_atoms=n_atoms, 
#                                 PBC=False,
#                                 real_cell=10, # fake
#                                 scaled_coords=False,
#                                 mode='continuous',
#                                 switching_function=switching_function)
    
#     out = model(pos)
#     out.sum().backward()
#     print(out)

def test_coordination_number():
    from mlcolvar.core.transform.tools.switching_functions import SwitchingFunctions

    # pos = torch.Tensor([[3.057573, -0.970874, 0.010963],
    #                     [3.303463, -0.921235, -0.093758],
    #                     [3.143809, -0.969587, 0.243834],
    #                     [3.144826, -1.055678, 1.234909],
    #                     [2.817620, -0.882992, 0.701040],
    #                     [3.523643, -0.600038, 1.250236],
    #                     [3.047862, -0.612398, 0.731384],
    #                     [3.029544, -0.438634, 1.131293],
    #                     [3.058277, -1.465733, 1.020088],
    #                     [3.267026, -1.142289, 0.492955]])
    
    
    pos = torch.Tensor([[-0.410219, -0.680065, -2.016121],
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
                        [-0.527323, -0.206236, -1.532587]])


    pos = torch.Tensor([[-0.410387, -0.677657, -2.018355],
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
                        [-0.528576, -0.202031, -1.534733]])
    
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

    # pos = torch.Tensor([[-0.186334, -0.576215, -2.052373],
    #                     [ 0.039420, -0.635731, -2.193494],
    #                     [-0.035417, -0.031627, -1.686382],
    #                     [-0.150362, -0.453559, -1.576238],
    #                     [ 0.049959, -0.267481, -1.556228],
    #                     [-0.401991, -0.700807, -1.459524],
    #                     [-0.649622, -0.853509, -1.617916],
    #                     [-0.291217, -0.244090, -1.488739],
    #                     [-0.029036, -0.804894, -1.580041],
    #                     [ 0.090713, -1.018815, -1.676822],
    #                     [-1.024799,  1.815174, -0.125380],
    #                     [-0.372569, -0.792924, -1.721617],
    #                     [-0.673611, -0.309695, -1.654188],
    #                     [ 0.183207, -0.407987, -1.758594],
    #                     [-0.464715, -0.479660, -1.722147],
    #                     [-0.389905, -0.116370, -1.718337]])
    cell = 4.0273098

    pos.requires_grad = True
    
    n_atoms = 12
    cutoff=0.25

    switching_function=SwitchingFunctions(in_features=n_atoms*3, name='Rational', cutoff=cutoff, options={'n': 2, 'm' : 6, 'eps' : 1e0})

    model = CoordinationNumbers(group_A = [0, 1],
                                group_B = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                                cutoff= cutoff,
                                n_atoms=n_atoms, 
                                PBC=True,
                                real_cell=cell,
                                scaled_coords=False,
                                mode='continuous',
                                switching_function=switching_function)
    
    out = model(pos)
    out.sum().backward()
    print(out)

if __name__ == "__main__":
    test_coordination_number()
        
