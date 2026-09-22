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