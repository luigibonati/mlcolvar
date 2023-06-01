import torch
from typing import Union
from warnings import warn

__all__ = ["Statistics", "Inverse"]


class Statistics(object):
    """
    Calculate statistics (running mean and std.dev based on Welford's algorithm, as well as min and max).
    If used with an iterable (such as a dataloader) provides the running estimates.
    To get the dictionary with the results use the .to_dict() method.
    """

    def __init__(self, X: torch.Tensor = None):
        self.count = 0

        self.properties = ["mean", "std", "min", "max"]

        # initialize properties and temp var M2
        for prop in self.properties:
            setattr(self, prop, None)
        setattr(self, "M2", None)

        self.__call__(X)

    def __call__(self, x):
        self.update(x)

    def update(self, x):
        if x is None:
            return

        # get batch size
        ndim = x.ndim
        if ndim == 0:
            x = x.reshape(1, 1)
        elif ndim == 1:
            x = x.unsqueeze(1)

        batch_size = x.shape[0]
        nfeatures = x.shape[1]

        new_count = self.count + batch_size

        # Initialize
        if self.mean is None:
            for prop in ["mean", "M2", "std"]:
                setattr(self, prop, torch.zeros(nfeatures))

        # compute sample mean
        sample_mean = torch.mean(x, dim=0)
        sample_m2 = torch.sum((x - sample_mean) ** 2, dim=0)

        # update stats
        delta = sample_mean - self.mean
        self.mean += delta * batch_size / new_count
        corr = batch_size * self.count / new_count
        self.M2 += sample_m2 + delta**2 * corr
        self.count = new_count
        self.std = torch.sqrt(self.M2 / self.count)

        # compute min/max
        sample_min = torch.min(x, dim=0).values
        sample_max = torch.max(x, dim=0).values

        if self.min is None:
            self.min = sample_min
            self.max = sample_max
        else:
            self.min = torch.min(torch.stack((sample_min, self.min)), dim=0).values
            self.max = torch.max(torch.stack((sample_max, self.max)), dim=0).values

    def to_dict(self) -> dict:
        return {prop: getattr(self, prop) for prop in self.properties}

    def __repr__(self):
        repr = "<Statistics>  "
        for prop in self.properties:
            repr += f"{prop}: {getattr(self,prop).numpy()} "
        return repr


class Inverse(torch.nn.Module):
    "Wrapper to return the inverse method of a module as a torch.nn.Module"

    def __init__(self, module: torch.nn.Module):
        """Return the inverse method of a module as a torch.nn.Module

        Parameters
        ----------
        module : torch.nn.Module
            Module to be inverted
        """
        super().__init__()
        if not hasattr(module, "inverse"):
            raise AttributeError("The given module does not have a 'inverse' method!")
        self.module = module

    def inverse(self, *args, **kwargs):
        return self.module.inverse(*args, **kwargs)

    def forward(self, *args, **kwargs):
        return self.inverse(*args, **kwargs)


def batch_reshape(t: torch.Tensor, size: torch.Size) -> torch.Tensor:
    """Return value reshaped according to size.
    In case of batch unsqueeze and expand along the first dimension.
    For single inputs just pass.

    Parameters
    ----------
        mean and range

    """
    if len(size) == 1:
        return t
    if len(size) == 2:
        batch_size = size[0]
        x_size = size[1]
        t = t.unsqueeze(0).expand(batch_size, x_size)
    else:
        raise ValueError(
            f"Input tensor must of shape (n_features) or (n_batch,n_features), not {size} (len={len(size)})."
        )
    return t


# ================================================================================================
# ======================================== TEST FUNCTIONS ========================================
# ================================================================================================


def test_statistics():
    # create fake data
    X = torch.arange(0, 100)
    X = torch.stack([X + 0.0, X + 100.0, X - 1000.0], dim=1)
    y = X.square().sum(1)
    print("X", X.shape)
    print("y", y.shape)

    # compute stats

    stats = Statistics()
    stats(X)
    print(stats)
    stats.to_dict()

    # create dataloader
    from mlcolvar.data import DictLoader

    loader = DictLoader({"data": X, "target": y}, batch_size=20)

    # compute statistics of a single key of loader
    key = "data"
    stats = Statistics()
    for batch in loader:
        stats.update(batch[key])
    print(stats)

    # compute stats of all keys in dataloader

    # init a statistics object for each key
    stats = {}
    for batch in loader:
        for key in loader.keys:
            # initialize
            if key not in stats:
                stats[key] = Statistics(batch[key])
            # or accumulate
            else:
                stats[key].update(batch[key])

    for key in loader.keys:
        print(key, stats[key])


if __name__ == "__main__":
    test_statistics()
