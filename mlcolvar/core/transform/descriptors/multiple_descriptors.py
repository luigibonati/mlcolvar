import torch

__all__ = ["MultipleDescriptors"]

class MultipleDescriptors(torch.nn.Module):
    """Wrapper class to combine multiple descriptor transform objects acting on the same set of atomic positions in a single module"""
    def __init__(self,
                 descriptors_list: list,
                 n_atoms: int,
                 ):
        """_summary_

        Parameters
        ----------
        descriptors_list : list
            List of descriptor transform objects to be combined
        n_atoms : int
            Number of atoms in the system
        """
        super().__init__()
        self.in_features = n_atoms * 3
        self.descriptors_list = descriptors_list

        self.out_features = 0
        for d in self.descriptors_list:
            self.out_features += d.out_features

    def forward(self, pos):
        for i,d in enumerate(self.descriptors_list):
            if i == 0:
                out = d(pos)
            else:
                aux = d(pos)
                out = torch.concatenate((out, aux), 1)
        return out

def test_multipledescriptors():
    from .torsional_angle import TorsionalAngle
    from .pairwise_distances import PairwiseDistances

    # check using torsional angles and distances in alanine
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
    cell = torch.Tensor([3.0233, 3.0233, 3.0233])

    # model 1 and 2 for torsional angles, model 3 for distances
    model_1 = TorsionalAngle([1,3,4,6], 10, ['angle'], False, cell, False)
    model_2 = TorsionalAngle([3,4,6,8], 10, ['angle'], False, cell, False)
    model_3 = PairwiseDistances(n_atoms = 10,
                                PBC = True,
                                cell = cell,
                                scaled_coords = False,
                                slicing_pairs=[[0, 1], [0, 2]])
    # compute single references
    angle_1 = model_1(pos)
    angle_2 = model_2(pos)
    distances = model_3(pos)

    # stack torsional angles
    model_tot = MultipleDescriptors([model_1, model_2], n_atoms = 10)
    out = model_tot(pos)
    out.sum().backward()
    for i in range(len(pos)):
        assert(torch.allclose(out[i, 0], angle_1[i]))
        assert(torch.allclose(out[i, 1], angle_2[i]))

    # stack torsional angle and two distances
    model_tot = MultipleDescriptors([model_1, model_3], n_atoms = 10)
    out = model_tot(pos)
    out.sum().backward()
    for i in range(len(pos)):
        assert(torch.allclose(out[i, 0], angle_1[i]))
        assert(torch.allclose(out[i, 1:], distances[i]))