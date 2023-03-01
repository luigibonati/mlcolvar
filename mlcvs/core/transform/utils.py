import torch

__all__ = ["RunningStats"]

class RunningStats(object):
    """
    Calculate statistics (running mean and std.dev based on Welford's algorithm, as well as min and max).
    """
    def __init__(self, X : torch.Tensor = None):
        self.count = 0

        self.properties = ['Mean','Std','Min','Max']

        # initialize properties and temp var M2
        for prop in self.properties:
            setattr(self,prop,None)
        setattr(self,'M2',None)
    
        self.__call__(X)

    def __call__(self,x):
        self.update(x)

    def update(self,x):
        if x is None:
            return 
        
        # get batch size
        ndim = x.ndim
        batch_size = 1
        if ndim == 0:
            x = x.reshape(1,1)
        elif ndim == 1:
            x = x.unsqueeze(1)
        
        batch_size = x.shape[0]
        nfeatures = x.shape[1]
        
        new_count = self.count + batch_size

        # Initialize
        if self.Mean is None:
            for prop in ['Mean','M2','Std']:
                setattr(self,prop, torch.zeros(nfeatures))

        # compute sample mean
        sample_mean = torch.mean(x, dim=0)
        sample_m2 = torch.sum((x - sample_mean) ** 2, dim=0)

        # update stats
        delta = sample_mean - self.Mean 
        self.Mean += delta * batch_size / new_count
        corr = batch_size * self.count / new_count
        self.M2 += sample_m2 + delta**2 * corr
        self.count = new_count
        self.Std = torch.sqrt(self.M2 / self.count)

        # compute min/max
        sample_min = torch.min(x, dim=0).values
        sample_max = torch.max(x, dim=0).values

        if self.Min is None:
            self.Min = sample_min
            self.Max = sample_max
        else:
            self.Min = torch.min( torch.stack((sample_min,self.Min)), dim=0).values
            self.Max = torch.max( torch.stack((sample_max,self.Max)), dim=0).values

    def to_dict(self) -> dict:
        return {prop: getattr(self,prop) for prop in self.properties}
    
    def __repr__(self):
        repr = "<Statistics>  " 
        for prop in self.properties:
            repr+= f"{prop}: {getattr(self,prop).numpy()} "
        return repr
  
def test_runningstats():
    # create fake data
    X = torch.arange(0,100)
    X = torch.stack([X+0.,X+100.,X-1000.],dim=1)
    y = X.square().sum(1)
    print('X',X.shape)
    print('y',y.shape)

    # compute stats
    
    stats = RunningStats()
    stats(X)
    print(stats)
    stats.to_dict()

    # create dataloader
    from mlcvs.data import FastDictionaryLoader
    loader = FastDictionaryLoader({'data':X,'target':y},batch_size=20)

    # compute statistics of a single key of loader
    key = 'data'
    stats = RunningStats()
    for batch in loader:
        stats.update(batch[key])
    print(stats)

    # compute stats of all keys in dataloader

    # init a runningstats object for each key
    stats = {}
    for batch in loader:
        for key in loader.keys:
            #initialize
            if key not in stats:
                stats[key] = RunningStats(batch[key])
            # or accumulate
            else:
                stats[key].update(batch[key])
        
    for key in loader.keys:
        print(key,stats[key])


def batch_reshape(t: torch.Tensor, size : torch.Size) -> (torch.Tensor):
    """Return value reshaped according to size. 
    In case of batch expand unsqueeze and expand along the first dimension.
    For single inputs just pass:

    Parameters
    ----------
        Mean and range 

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


