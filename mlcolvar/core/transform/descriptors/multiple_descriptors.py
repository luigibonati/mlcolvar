import torch
import inspect

__all__ = ["MultipleDescriptors"]

class MultipleDescriptors(torch.nn.Module):
    """Wrapper class to combine multiple descriptor transform objects acting on the same set of atomic positions"""
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
        # Use ModuleList instead of plain list to ensure proper device handling
        self.descriptors_list = torch.nn.ModuleList(descriptors_list)

        self.out_features = 0
        for d in self.descriptors_list:
            self.out_features += d.out_features

    @property
    def device(self):
        """Check device consistency and return device"""
        devices = set()

        # Check module's parameters
        for p in self.parameters(recurse=False):
            devices.add(p.device)

        # Check module's buffers
        for b in self.buffers(recurse=False):
            devices.add(b.device)

        # Check submodules
        for d in self.descriptors_list:
            for p in d.parameters():
                devices.add(p.device)
            for b in d.buffers():
                devices.add(b.device)

        if len(devices) == 0:
            return torch.device("cpu")

        if len(devices) > 1:
            raise RuntimeError(
                f"Inconsistent devices detected in module: {devices}"
            )

        return next(iter(devices))

    @staticmethod
    def _apply_descriptor(descriptor, pos, cell=None):
        if cell is None:
            return descriptor(pos)
        signature = inspect.signature(descriptor.forward)
        if "cell" in signature.parameters:
            return descriptor(pos, cell=cell)
        return descriptor(pos)

    def forward(self, pos, cell=None):
        # move input to model's device
        if isinstance(pos, torch.Tensor):
            model_device = self.device
            if pos.device != model_device:
                pos = pos.to(model_device)
        
        for i,d in enumerate(self.descriptors_list):
            if i == 0:
                out = self._apply_descriptor(d, pos, cell=cell)
            else:
                aux = self._apply_descriptor(d, pos, cell=cell)
                out = torch.concatenate((out, aux), 1)
        return out

def test_multipledescriptors():
    from .torsional_angles import TorsionalAngles
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
    model_1 = TorsionalAngles(indices=[1,3,4,6], n_atoms=10, mode=['angle'], PBC=False, cell=cell, scaled_coords=False)
    model_2 = TorsionalAngles(indices=[3,4,6,8], n_atoms=10, mode=['angle'], PBC=False, cell=cell, scaled_coords=False)
    model_3 = PairwiseDistances(n_atoms=10, PBC=True, cell=cell, scaled_coords=False, slicing_pairs=[[0, 1], [0, 2]])
    
    # compute single references
    angle_1 = model_1(pos)
    angle_2 = model_2(pos)
    distances = model_3(pos)

    # stack torsional angles
    model_tot = MultipleDescriptors(descriptors_list=[model_1, model_2], n_atoms=10)
    out = model_tot(pos)
    out.sum().backward()
    for i in range(len(pos)):
        assert(torch.allclose(out[i, 0], angle_1[i]))
        assert(torch.allclose(out[i, 1], angle_2[i]))

    # stack torsional angle and two distances
    model_tot = MultipleDescriptors(descriptors_list=[model_1, model_3], n_atoms=10)
    out = model_tot(pos)
    out.sum().backward()
    for i in range(len(pos)):
        assert(torch.allclose(out[i, 0], angle_1[i]))
        assert(torch.allclose(out[i, 1:], distances[i]))

    # ---------------- test varying-cell case (batched cells) ----------------
    # Mix different cell sizes in the same batch.
    scales = torch.tensor([0.9, 1.0, 1.1], dtype=pos.dtype)
    pos_base = pos.clone().detach()
    pos_batched = torch.cat([pos_base * s for s in scales], dim=0).clone().detach().requires_grad_(True)
    frame_scales = scales.repeat_interleave(pos_base.shape[0])
    cell_batched = torch.stack([cell * s for s in frame_scales], dim=0)

    # Rebuild descriptors with dynamic cell input.
    model_1 = TorsionalAngles(indices=[1,3,4,6], n_atoms=10, mode=['angle'], PBC=False, cell=None, scaled_coords=False)
    model_2 = TorsionalAngles(indices=[3,4,6,8], n_atoms=10, mode=['angle'], PBC=False, cell=None, scaled_coords=False)
    model_3 = PairwiseDistances(n_atoms=10, PBC=True, cell=None, scaled_coords=False, slicing_pairs=[[0, 1], [0, 2]])

    # stack torsional angles only: invariance under uniform scaling
    model_tot = MultipleDescriptors(descriptors_list=[model_1, model_2], n_atoms=10)
    out = model_tot(pos_batched, cell=cell_batched)
    out_single = torch.cat(
        [model_tot(pos_batched[i:i+1], cell=cell_batched[i]) for i in range(pos_batched.shape[0])],
        dim=0,
    )
    assert(torch.allclose(out, out_single, atol=1e-6))
    ref_torsions = model_tot(pos_base, cell=cell)
    ref_torsions_batched = torch.cat([ref_torsions for _ in scales], dim=0)
    assert(torch.allclose(out, ref_torsions_batched, atol=1e-3))
    out.sum().backward()

    # stack torsional angle + pairwise distances
    model_tot = MultipleDescriptors(descriptors_list=[model_1, model_3], n_atoms=10)
    out = model_tot(pos_batched, cell=cell_batched)
    out_single = torch.cat(
        [model_tot(pos_batched[i:i+1], cell=cell_batched[i]) for i in range(pos_batched.shape[0])],
        dim=0,
    )
    assert(torch.allclose(out, out_single, atol=1e-6))

    # Torsional part invariant, pairwise-distance part scales with the cell.
    ref_mix = model_tot(pos_base, cell=cell)
    ref_mix_batched = []
    for s in scales:
        ref_mix_batched.append(
            torch.cat([ref_mix[:, :1], ref_mix[:, 1:] * s], dim=1)
        )
    ref_mix_batched = torch.cat(ref_mix_batched, dim=0)
    assert(torch.allclose(out, ref_mix_batched, atol=2e-3))
    out.sum().backward()
