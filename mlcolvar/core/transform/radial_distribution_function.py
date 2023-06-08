import torch

from mlcolvar.core.transform import Transform,PairwiseDistances
from mlcolvar.core.transform.utils import easy_KDE

from typing import Union

__all__ = ["RDF"]

class RDF(Transform):
    """
    Compute radial distribution function. Optionally returns the integral over some specific peaks.
    """

    def __init__(self,
                 n_atoms : int,
                 PBC : bool,
                 real_cell : torch.Tensor,
                 scaled_coords : bool,
                 min_max : list,
                 n_bins : int,
                 sigma_to_center : float,
                 peak_integral : list = None,
                 integral_baseline : bool = False) -> torch.Tensor :
        # check output dimension
        if peak_integral:
            super().__init__(in_features=int(n_atoms*(n_atoms-1) / 2), out_features=1)
        else:
            super().__init__(in_features=int(n_atoms*(n_atoms-1) / 2), out_features=n_bins)
        
        # parse arguments
        self.in_features = int(n_atoms*(n_atoms-1) / 2)
        self.n_atoms = n_atoms
        self.PBC = PBC
        self.scaled_coords = scaled_coords

        if real_cell.shape[0] != 1:
           if real_cell[0] == real_cell[1] and real_cell[1] == real_cell[2]:
                pass
           else:
                raise ValueError('Radial distribution function implemented only for cubic cells!') 
        self.real_cell = real_cell

        self.min_max = min_max
        self.n_bins = n_bins
        self.sigma_to_center = sigma_to_center

        self.peak_integral = peak_integral
        self.integral_baseline = integral_baseline

        # initialize pairwise distances transform
        self.pairDist = PairwiseDistances(n_atoms=self.n_atoms,
                                 PBC=self.PBC,
                                 real_cell = self.real_cell,
                                 scaled_coords=self.scaled_coords)
        
    def compute_distances(self, x):
        dist = self.pairDist(x)
        return dist
    
    def compute_hist(self, x):
        hist, bins = easy_KDE(x=x,
                              n_input=self.in_features, 
                              min_max=self.min_max, 
                              n=self.n_bins, 
                              sigma_to_center=self.sigma_to_center,
                              return_bins=True)
        
        # store the bins, shell and integration attributes if not present
        if not hasattr(self, 'bins'):
            self.bins = bins
            self.bins_size = bins[1] - bins[0]

            bins_ext = torch.linspace(self.min_max[0], self.min_max[1] + self.bins_size.item(), self.n_bins + 1, device=x.device)
            self.shell = 4/3 * torch.pi * (self.n_atoms / self.real_cell**3) * (bins_ext[1:]**3 - bins_ext[:-1]**3)

            if self.peak_integral is not None:
                self.peak_lower_bound =  torch.where(self.bins == torch.max(self.bins[(self.bins < self.peak_integral[0])]))[0].item()
                self.peak_upper_bound =  torch.where(self.bins == torch.min(self.bins[(self.bins > self.peak_integral[1])]))[0].item()

        return hist
    
    def compute_rdf(self, x):
        x = self.compute_distances(x)
        x = self.compute_hist(x)
        x = x / (self.shell*self.n_atoms)
        return x
        

    def forward(self, x: torch.Tensor):
        x = self.compute_rdf(x)
        if self.peak_integral is not None:
            if self.integral_baseline:
                integral = torch.sum((x[:, self.peak_lower_bound:self.peak_upper_bound] - 1) * self.bins_size, dim = 1, keepdim=True)
            else:
                integral = torch.sum((x[:, self.peak_lower_bound:self.peak_upper_bound]) * self.bins_size, dim = 1, keepdim=True)
            return integral
        else:
            return x

def test_radial_distribution_function():
    x = torch.randn((5,300))
    cell = 1.0
    rdf = RDF(n_atoms=100,
              PBC=True,
              real_cell=cell,
              scaled_coords=False,
              min_max=[0,1],
              n_bins=10,
              sigma_to_center=1,
              peak_integral=[0.2,0.8],
              integral_baseline = True)
    out = rdf(x)
    
if __name__ == "__main__":
    test_radial_distribution_function()