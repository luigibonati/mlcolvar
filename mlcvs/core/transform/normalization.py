import torch
import pytorch_lightning as pl
from mlcvs.core.transform.utils import batch_reshape

class RunningAverageStd(object):
    """
    Implements Welford's algorithm for computing a running mean
    and standard deviation as described at: 
        http://www.johndcook.com/standard_deviation.html

    Adapted for pytorch from: https://gist.github.com/alexalemi/2151722

    can take single values or iterables

    Attributes
    ----------
    mean : 
        returns the mean
    std : int
        returns the std
    range: 
        alias for std
    meanfull: 
        returns the mean and std of the mean
    """

    def __init__(self,lst=None):
        self.k = 0
        self.M = None
        self.S = None
        
        self.__call__(lst)
    
    def update(self,x):
        if x is None:
            return
        self.k += 1
        if self.M is not None:
            newM = self.M + (x - self.M)*1./self.k
            newS = self.S + (x - self.M)*(x - newM)
        else: 
            newM = x
            newS = x * 0.
        self.M, self.S = newM, newS

    def consume(self,lst):
        lst = iter(lst)
        for x in lst:
            self.update(x)
    
    def reset(self):
        self.k = 0
        self.M = None
        self.S = None

    def __call__(self,x):
        if hasattr(x,"__iter__"):
            self.consume(x)
        else:
            self.update(x)
            
    @property
    def mean(self):
        return self.M
    @property
    def meanfull(self):
        return self.mean, self.std/torch.sqrt(self.k)
    @property
    def std(self):
        if self.k==0:
            return self.S
        if self.k==1:
            return 0
        return torch.sqrt(self.S/(self.k-1))
    @property
    def range(self):
        return self.std
    def __repr__(self):
        return f"<Running average +- std: {self.mean} +- {self.std}>"

class RunningMinMax(object):
    """
    Running calculation of min and max.

    can take single values or iterables

    Attributes
    ----------
    min : 
        returns the min
    max : int
        returns the max
    mean: 
        returns the mean (min+max)/2
    range: 
        returns the range (max-min)
    """

    def __init__(self,lst=None):
        self.k = 0 
        self.Min = None
        self.Max = None
        self.M = None
        self.S = None
        
        self.dim = 0
        self.__call__(lst)
    
    def update(self,x):
        if x is None:
            return
        self.k += 1
        if self.M is not None:
            min_stack = torch.stack((torch.min(x,self.dim)[0], self.Min )) 
            newMin = torch.min(min_stack,dim=self.dim)[0]
            max_stack = torch.stack((torch.max(x,self.dim)[0], self.Max )) 
            newMax = torch.max(max_stack,dim=self.dim)[0]
        else: 
            newMin = torch.min(x,dim=self.dim).values
            newMax = torch.max(x,dim=self.dim).values

        self.Min, self.Max = newMin, newMax
        self.M = (self.Max + self.Min) / 2.0
        self.S = (self.Max - self.Min) / 2.0
    
    def reset(self):
        self.k = 0
        self.Min = None
        self.Max = None
        self.M = None
        self.S = None

    def __call__(self,x):
        if (x is not None) and (x.ndim == 0):
            x = x.unsqueeze(0) 
        self.update(x)
            
    @property
    def mean(self):
        return self.M
    @property
    def range(self):
        return self.S
    def __repr__(self):
        return f"<Running mean and range : {self.mean}, {self.range} (min and max: [{self.Min},{self.Max}])>"

class Normalization(torch.nn.Module):
    """Normalizing block, used for computing standardized inputs/outputs.
    By default it accumulates statistics from a single epoch and then save the parameters. This can be changed with the hooks dictionary.
    """

    def __init__(self, n_in : int, mode : str = 'mean_std', epoch_wise : bool = True):
        """Initialize a normalization object. 

        The standardization mode can be either 'mean_std' (remove by the mean and divide by the standard deviation) or 'min_max' (scale and shift the range of values such that all inputs are between -1 and 1).

        By default the normalization is computed on a running estimate based on each epoch. Disable it to save the parameters at the batch level. 

        Parameters
        ----------
        n_in : int
            number of inputs
        mode : str, optional
            normalization mode (mean_std, min_max), by default 'mean_std'
        epoch_wise : bool, optional
            whether to accumulate statistics along each epoch or simply saving parameters at batch level , by default True
        """

        super().__init__()
        
        # buffers containing mean and range for standardization
        self.register_buffer("Mean", torch.zeros(n_in))
        self.register_buffer("Range", torch.ones(n_in))

        self.n_in = n_in
        self.n_out = n_in

        self.mode = mode

        if self.mode == 'mean_std':
            self.running_stats = RunningAverageStd()
        elif self.mode == 'min_max':
            self.running_stats = RunningMinMax()
        else:
            raise ValueError(f'The normalization mode should be one of the following: "mean_std", "min_max", not {mode}.')

        self.epoch_wise = epoch_wise

    def extra_repr(self) -> str:
        return f"n_in={self.n_in}, mode={self.mode}, epoch_wise={self.epoch_wise}"

    def reset_stats(self) -> None:
        """Reset running statistics."""
        self.running_stats.reset()

    @torch.no_grad()
    def save_stats(self) -> None:
        """Save running statistics into class methods for later inference."""
        self.Mean = self.running_stats.mean.detach()
        self.Range = self.running_stats.range.detach()

    def on_train_epoch_start(self) -> None:
        if self.epoch_wise:
            self.reset_stats()

    def on_train_epoch_end(self) -> None:
        if self.epoch_wise:
            self.save_stats()

    def batch_reshape(t: torch.Tensor, size : torch.Size) -> (torch.Tensor):
        """Return value reshaped according to size. 
        In case of batch expand unsqueeze and expand along the first dimension.
        For single inputs just pass:

        Parameters
        ----------
            Mean and range 

        """
        if len(size) == 1:
            pass
        if len(size) == 2:
            batch_size = size[0]
            x_size = size[1]
            t = t.unsqueeze(0).expand(batch_size, x_size)
        else:
            raise ValueError(
                f"Input tensor must of shape (n_features) or (n_batch,n_features), not {size}."
            )
        return t

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
        # accumulate statistics during training
        if self.training:
            self.running_stats(x)
            if not self.epoch_wise: # ==> save and reset batch-wise
                self.save_stats()
                self.reset_stats()
            
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


def test_running_average():

    foo = RunningAverageStd()
    foo(torch.arange(0, 50 ))
    print(foo)
    foo(torch.arange(51,100))
    print(foo)
    assert foo.mean.ndim == 0

    foo = RunningAverageStd()
    foo(torch.arange(0, 50).reshape(10,5))
    print(foo)
    assert (foo.mean.ndim == 1 ) and ( foo.std.shape[0] == 5)
    foo.reset()

def test_running_min_max():

    print('-------------')
    foo = RunningMinMax()
    foo(torch.arange(0, 50 ))
    print(foo)
    foo(torch.arange(51,100))
    print(foo)
    assert foo.mean.ndim == 0
    print('-------------')
    foo = RunningMinMax()
    foo(torch.arange(0, 50).reshape(10,5))
    print(foo)
    assert (foo.mean.ndim == 1 ) and ( foo.range.shape[0] == 5)
    foo.reset()
    print('-------------')
    foo = RunningMinMax()
    X = torch.Tensor([[1,100,],[10,2],[5,50]])
    foo(X)
    print(foo)


def test_normalization():
    torch.manual_seed(42)

    n_in = 2
    norm = Normalization(n_in, mode='mean_std')

    X = torch.randn((5,n_in))*10

    # accumulate stats during training
    print('TRAIN')
    norm.train()
    # reset variables 
    norm.reset_stats()
    # propagate X and accumulate
    print(norm(X))
    # display current stats
    print('running stats')
    print(norm.running_stats)
    # save running stats for inference
    norm.save_stats()
    print('saved stats')
    print(norm.Mean,norm.Range)

    # Then use the estimate for predict
    print('EVAL')
    norm.eval()
    print('Normalize')
    print(norm(X))
    print('Un-normalize')
    print(norm.inverse(norm(X)))
    print('-------------')
    # check other way of normalizing
    norm = Normalization(n_in, mode='min_max')
    norm.train()
    X = torch.Tensor([[1,100,],[10,2],[5,50]])
    print('input')
    print(X)
    norm(X)
    norm.save_stats()
    print('min', norm.running_stats.Min)
    print('max', norm.running_stats.Max)
    assert torch.allclose( norm.running_stats.Min, X.min(0)[0])
    assert torch.allclose( norm.running_stats.Max, X.max(0)[0])
    print(norm)

if __name__ == "__main__":
    test_running_average()
    test_running_min_max()
    test_normalization()