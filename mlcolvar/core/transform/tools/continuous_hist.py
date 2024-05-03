import torch

from mlcolvar.core.transform import Transform
from mlcolvar.core.transform.tools.utils import easy_KDE

__all__ = ["ContinuousHistogram"]

class ContinuousHistogram(Transform):
    """
    Compute continuous histogram using Gaussian kernels
    """

    def __init__(self,
                 in_features : int,
                 min : float,
                 max : float,
                 bins : int,
                 sigma_to_center : float = 1.0) -> torch.Tensor :
        """Computes the continuous histogram of a quantity using Gaussian kernels

        Parameters
        ----------
        in_features : int
            Number of inputs
        min : float
            Minimum value of the histogram
        max : float
            Maximum value of the histogram
        bins : int
            Number of bins of the histogram
        sigma_to_center : float, optional
            Sigma value in bin_size units, by default 1.0


        Returns
        -------
        torch.Tensor
            Values of the histogram for each bin
        """
       
        super().__init__(in_features=in_features, out_features=bins)

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
    x.requires_grad = True
    hist = ContinuousHistogram(in_features=100,
                    min=-1,
                    max=1,
                    bins=10,
                    sigma_to_center=1)
    out = hist(x)
    out.sum().backward()
    
if __name__ == "__main__":
    test_continuous_histogram()