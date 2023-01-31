import torch
import pytorch_lightning as pl
from mlcvs.utils.decorators import decorate_methods, allowed_hooks, apply_hooks

@decorate_methods(apply_hooks,methods=allowed_hooks)
class Normalization(pl.LightningModule):
    """Normalizing block, used for computing standardized inputs/outputs.
    By default it accumulates statistics from a single epoch and then save the parameters. This can be changed with the hooks dictionary.
    """

    def __init__(self, n_in : int, mode : str = 'std', hooks : dict = {'on_train_epoch_end': 'save_stats', 'on_train_epoch_start': 'reset_stats' }):
        """Initialize a normalization object. This will accumulate statistics and then save the parameters. 
        The time when these the reset/save parameters action are performed can be changed via the hooks dictionary.

        The standardization mode can be either 'std' (remove by the mean and divide by the standard deviation) or 'minmax' (scale and shift the range of values such that all inputs are between -1 and 1).

        Parameters
        ----------
        n_in : int
            number of inputs
        mode : str, optional
            normalization mode (std,minmax), by default 'std'
        hooks : dict, optional
            dictionary which specify when to execute reset/save of accumulated parameters, by default {'on_train_epoch_end': 'save_stats', 'on_train_epoch_start': 'reset_stats' }

        """

        super().__init__()
        
        # buffers containing mean and range for standardization
        self.register_buffer("Mean", torch.zeros(n_in))
        self.register_buffer("Range", torch.ones(n_in))

        self.n_in = n_in
        self.n_out = n_in

        if mode == 'std':
            self.running_stats = RunningAverageStd()
        elif mode == 'minmax':
            self.running_stats = RunningMinMax()
        else:
            raise ValueError(f'The normalization mode should be one of the following: "std", "minmax", not {mode}.')

        self.hooks = hooks

    def reset_stats(self) -> None:
        """Reset running statistics."""
        self.running_stats.reset()

    @torch.no_grad()
    def save_stats(self) -> None:
        """Save running statistics into class methods for later inference."""
        self.Mean = self.running_stats.mean.detach()
        self.Range = self.running_stats.range.detach()

    def get_mean_range(self, size : torch.Size) -> (torch.Tensor):
        """Return mean and range reshaped according to tensor size. 

        Parameters
        ----------
        size : torch.Size
        
        Returns
        -------
        (Mean, Range): Tuple[torch.Tensor]
            Mean and range 

        """
        if len(size) == 2:
            batch_size = size[0]
            x_size = size[1]
            Mean = self.Mean.unsqueeze(0).expand(batch_size, x_size)
            Range = self.Range.unsqueeze(0).expand(batch_size, x_size)
        elif len(size) == 1:
            Mean = self.Mean
            Range = self.Range
        else:
            raise ValueError(
                f"Input tensor must of shape (n_features) or (n_batch,n_features), not {size}."
            )
        return (Mean,Range)

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
        # get mean and range
        Mean,Range = self.get_mean_range(x.size())
    
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
        Mean,Range = self.get_mean_range(x.size())

        return x.mul(Range).add(Mean)

import torch

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
    norm = Normalization(n_in, mode='std')

    X = torch.randn((5,n_in))*10

    # accumulate stats during training
    print('TRAIN')
    norm.train()
    # reset variables 
    norm.on_train_epoch_start()
    #norm.reset_stats()
    # propagate X and accumulate
    print(norm(X))
    # display current stats
    print('running stats')
    print(norm.running_stats)
    # save running stats for inference
    norm.on_train_epoch_end()
    #norm.save_stats()
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
    norm = Normalization(n_in, mode='minmax')
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

if __name__ == "__main__":
    test_running_average()
    test_running_min_max()
    test_normalization()