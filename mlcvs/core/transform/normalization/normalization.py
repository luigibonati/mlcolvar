import torch
import pytorch_lightning as pl
from mlcvs.core.utils.decorators import decorate_methods, allowed_hooks, apply_hooks
from .utils import RunningAverageStd,RunningMinMax



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
    test_normalization()