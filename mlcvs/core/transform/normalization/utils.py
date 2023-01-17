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

if __name__ == "__main__":
    test_running_average()
    test_running_min_max()
