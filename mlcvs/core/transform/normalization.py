import torch
from mlcolvar.core.transform.utils import batch_reshape,Statistics
from mlcolvar.core.transform import Transform

__all__ = ["Normalization"]

def sanitize_range( range : torch.Tensor):
    """Sanitize

    Parameters
    ----------
    range : torch.Tensor
        range to be used for standardization

    """

    if (range < 1e-6).nonzero().sum() > 0:
        print(
        "[Warning] Normalization: the following features have a range of values < 1e-6:",
        (range < 1e-6).nonzero(),
        )
    range[range < 1e-6] = 1.0

    return range


class Normalization(Transform):
    """
    Normalizing block, used for computing standardized inputs/outputs.
    """

    def __init__(self, in_features : int, mean : torch.Tensor = None, range : torch.Tensor = None, stats : dict = None, mode : str = 'mean_std'):
        """Initialize a normalization object. Values will be subtracted by self.mean and then divided by self.range.
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

        super().__init__(in_features=in_features, out_features=in_features)
        
        # buffers containing mean and range for standardization
        self.register_buffer("mean", torch.zeros(in_features))
        self.register_buffer("range", torch.ones(in_features))
        
        self.mode = mode
        self.is_initialized = False

        # set values based on args if provided
        self.set_custom(mean,range)
        if stats is not None:
            self.set_from_stats(stats,mode=mode)

        # save params
        self.in_features = in_features
        self.out_features = in_features

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, mode={self.mode}"

    def set_custom(self, mean : torch.Tensor = None, range : torch.Tensor = None): 
        """ Set parameter of the normalization layer. 

        Parameters
        ----------
        mean : torch.Tensor
            Value that will be removed.
        range : torch.Tensor, optional
            Value that will be divided for.
        """

        if mean is not None:
            self.mean = mean
        if range is not None:
            self.range = sanitize_range(range)

        if mean is not None or range is not None:
            self.is_initialized = True
            self.mode = 'custom'

    def set_from_stats(self, stats : dict or Statistics, mode : str = None): 
        """ Set parameters of the normalization layer based on a dictionary with statistics 

        Parameters
        ----------
        stats : dict or Statistics
            dictionary with statistics
        mode : str, optional
            standardization mode ('mean_std' or 'min_max'), by default None (will use self.mode)
        """

        if mode is None:
            mode = self.mode
        if isinstance(stats, Statistics):
            stats = stats.to_dict()

        if mode == 'mean_std':
            self.mean = stats['mean']
            range = stats['std'] 
            self.range = sanitize_range( range ) 
        elif mode == 'min_max':
            min = stats['min']
            max = stats['max']
            self.mean = (max + min) / 2.0
            range = (max - min) / 2.0
            self.range = sanitize_range( range ) 
        elif mode == 'custom':
            raise AttributeError('If mode is custom the parameters should be supplied via mean and range values when creating the Normalization object or with the set_custom, not with set_from_stats.')
        else: 
            raise ValueError(f'Mode {self.mode} unknonwn. Available modes: "mean_std", "min_max","custom"')
    
        self.is_initialized = True
        
        if mode != self.mode:
            self.mode = mode

    def setup_from_datamodule(self,datamodule):
        if not self.is_initialized:
            # obtain statistics from the dataloader
            try:
                stats = datamodule.train_dataloader().get_stats()['data']
            except KeyError:
                raise ValueError(f'Impossible to initialize {self.__class__.__name__} '
                                 'because the training dataloader does not have a "data" key '
                                 '(are you using multiple datasets?). A manual initialization '
                                 'of "mean" and "range" is necessary.')
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
        mean  = batch_reshape(self.mean, x.size())
        range = batch_reshape(self.range, x.size())

        return x.sub(mean).div(range)

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
        mean  = batch_reshape(self.mean, x.size())
        range = batch_reshape(self.range, x.size())

        return x.mul(range).add(mean)


def test_normalization():
    # create data
    torch.manual_seed(42)
    in_features = 2
    X = torch.randn((100,in_features))*10

    # get stats
    from mlcolvar.core.transform.utils import Statistics
    stats = Statistics(X).to_dict()
    norm = Normalization(in_features, mean=stats['mean'],range=stats['std'])

    y = norm(X)
    print(X.mean(0),y.mean(0))
    print(X.std(0),y.std(0))
    

if __name__ == "__main__":
    test_normalization()