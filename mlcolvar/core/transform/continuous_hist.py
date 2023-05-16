import torch

from mlcolvar.core.transform import Transform
from mlcolvar.core.transform.utils import easy_KDE

from typing import Union

__all__ = ["ContHist"]

class ContHist(Transform):
    """
    Compute continuous histogram with KDE-like method
    """

    def __init__(self,
                 in_features : int,
                 min : float,
                 max : float,
                 bins : int,
                 sigma_to_center : float) -> torch.Tensor :
       
        super().__init__(in_features=in_features, out_features=bins)

        self.in_features = in_features
        self.min = min
        self.max = max
        self.bins = bins
        self.sigma_to_center = sigma_to_center
    
    def compute_hist(self, x):
        hist = easy_KDE(x=x,
                        n_input=self.in_features, 
                        min_max=[self.min, self.max], 
                        n=self.bins, 
                        sigma_to_center=self.sigma_to_center)
        # if self.normalize = 
        return hist

    def forward(self, x: torch.Tensor):
        x = self.compute_hist(x)
        return x
    
def test_continuous_histogram():
    x = torch.randn((5,100))
    hist = ContHist(in_features=100,
                    min=-1,
                    max=1,
                    bins=10,
                    sigma_to_center=1)
    out = hist(x)
    
if __name__ == "__main__":
    test_continuous_histogram()