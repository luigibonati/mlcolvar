import torch
from mlcvs.core.transform.utils import batch_reshape
from mlcvs.core.transform import Transform

__all__ = ["Normalization"]

def sanitize_range( Range : torch.Tensor):
    """Sanitize

    Parameters
    ----------
    Range : torch.tensor
        range to be used for standardization

    """

    if (Range < 1e-6).nonzero().sum() > 0:
        print(
        "[Warning] Normalization: the following features have a range of values < 1e-6:",
        (Range < 1e-6).nonzero(),
        )
    Range[Range < 1e-6] = 1.0

    return Range


class Normalization(Transform):
    """
    Normalizing block, used for computing standardized inputs/outputs.
    """

    def __init__(self, in_features : int, mean : torch.Tensor = None, range : torch.Tensor = None, mode : str = 'mean_std'):
        """Initialize a normalization object. 
        The parameters for the standardization can be either given from the user (via mean/range keywords), or they can be calculated from a datamodule. 
        In the former, the mode will be overriden as 'custom'. 'In the latter, the standardization mode can be either 'mean_std' (remove by the mean and divide by the standard deviation) or 'min_max' (scale and shift the range of values such that all inputs are between -1 and 1).
                                                        
        Parameters
        ----------
        in_features : int
            number of inputs
        mean: torch.Tensor, optional
            values to be subtracted
        range: torch.Tensor, optional
            values to be scaled by        
        mode : str, optional 
            normalization mode (mean_std, min_max), by default 'mean_std'
        """

        super().__init__()
        
        # buffers containing mean and range for standardization
        self.register_buffer("Mean", torch.zeros(in_features))
        self.register_buffer("Range", torch.ones(in_features))
        
        self.mode = mode
        self.is_initialized = False

        # set mean and range if provided
        self.set_custom(mean,range)

        # save params
        self.in_features = in_features
        self.out_features = in_features

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, mode={self.mode}"

    def set_custom(self, mean : torch.Tensor = None, range : torch.Tensor = None): # TODO DOC

        if mean is not None:
            self.Mean = mean
        if range is not None:
            self.Range = sanitize_range(range)

        if mean is not None or range is not None:
            self.is_initialized = True
            self.mode = 'custom'

    def set_from_stats(self,stats : dict ,mode=None): # TODO DOC
        if mode is None:
            mode = self.mode

        if mode == 'mean_std':
            self.Mean = stats['Mean']
            Range = stats['Std'] 
            self.Range = sanitize_range( Range ) 
        elif mode == 'min_max':
            Min = stats['Min']
            Max = stats['Max']
            self.Mean = (Max + Min) / 2.0
            Range = (Max - Min) / 2.0
            self.Range = sanitize_range( Range ) 
        elif mode == 'custom':
            raise AttributeError('If mode is custom the parameters should be supplied when creating the Normalization object or with the set_custom, not with set_from_stats')
        else: 
            raise ValueError(f'Mode {self.mode} unknonwn. Available modes: "mean_std", "min_max","custom"')
    
        self.is_initialized = True
        
        if mode != self.mode:
            self.mode = mode

    def setup_from_datamodule(self,datamodule):
        if not self.is_initialized:
            # obtain statistics from the dataloader
            stats = datamodule.train_dataloader().get_stats()['data']
            self.set_from_stats(stats,self.mode)

    def forward(self, x: torch.Tensor) -> (torch.Tensor):
        """
        Compute standardized inputs.

        Parameters
        ----------
        x: torch.Tensor
            input/output

        Returns
        -------
        out : torch.Tensor
            standardized inputs
        """
            
        # get mean and range
        Mean  = batch_reshape(self.Mean, x.size())
        Range = batch_reshape(self.Range, x.size())

        return x.sub(Mean).div(Range)

    def inverse(self, x: torch.Tensor) -> (torch.Tensor):
        """
        Remove standardization.

        Parameters
        ----------
        x: torch.Tensor
            input

        Returns
        -------
        out : torch.Tensor
            un-normalized inputs
        """
        # get mean and range
        Mean  = batch_reshape(self.Mean, x.size())
        Range = batch_reshape(self.Range, x.size())

        return x.mul(Range).add(Mean)


def test_normalization():
    # create data
    torch.manual_seed(42)
    in_features = 2
    X = torch.randn((100,in_features))*10

    # get stats
    from mlcvs.core.transform.utils import RunningStats
    stats = RunningStats(X).to_dict()
    norm = Normalization(in_features, mean=stats['Mean'],range=stats['Std'])

    y = norm(X)
    print(X.mean(0),y.mean(0))
    print(X.std(0),y.std(0))

if __name__ == "__main__":
    test_normalization()