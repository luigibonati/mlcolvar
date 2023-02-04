import torch
import numpy as np
import pandas as pd
from bisect import bisect_left

__all__ = ['find_time_lagged_configurations','create_time_lagged_dataset']

def closest_idx(array, value):
        '''
        Find index of the element of 'array' which is closest to 'value'.
        The array is first converted to a np.array in case of a tensor.
        Note: it does always round to the lowest one.

        Parameters:
            array (tensor/np.array)
            value (float)

        Returns:
            pos (int): index of the closest value in array
        '''
        if type(array) is np.ndarray:
            pos = bisect_left(array, value)
        else:
            pos = bisect_left(array.numpy(), value)
        if pos == 0:
            return 0
        elif pos == len(array):
            return -1
        else:
            return pos-1

# evaluation of tprime from simulation time and logweights
def tprime_evaluation(t, logweights = None):
    """
    Estimate the accelerated time if a set of (log)weights is given

    Parameters
    ----------
    t : array-like, 
        unbias time series,    
    logweights : array-like,optional
        logweights to evaluate rescaled time as dt' = dt*exp(logweights)
    """

    # rescale time with log-weights if given
    if logweights is not None:
        # compute time increment in simulation time t
        dt = np.round(t[1]-t[0],5)
        # sanitize logweights
        logweights = torch.Tensor(logweights)
        # when the bias is not deposited the value of bias potential is minimum 
        logweights -= torch.max(logweights)
        # bug: exp(logweights/lognorm) != exp(logweights)/norm, where norm is sum_i beta V_i
        # are we changing the temperature? 
        """ possibilities:
            1) logweights /= torch.min(logweights) -> logweights belong to [0,1] 
            2) pass beta as an argument, then logweights *= beta
            3) tprime = dt * torch.cumsum( torch.exp( torch.logsumexp(logweights,0) ) ,0)
            4) tprime = dt *torch.exp ( torch.log (torch.cumsum (torch.exp(logweights) ) ) )
        """
        lognorm = torch.logsumexp(logweights,0)
        logweights /= lognorm
        # compute instantaneus time increment in rescaled time t'
        d_tprime = torch.exp(logweights)*dt
        # calculate cumulative time t'
        tprime = torch.cumsum(d_tprime,0)
    else:
        tprime = t

    return tprime

def find_time_lagged_configurations(x,t,lag_time):
    '''
    Searches for all the pairs which are distant 'lag' in time, and returns the weights associated to lag=lag as well as the weights for lag=0.

    Parameters:
        x (tensor): array whose columns are the descriptors and rows the time evolution
        time (tensor): array with the simulation time
        lag (float): lag-time

    Returns:
        x_t (tensor): array of descriptors at time t
        x_lag (tensor): array of descriptors at time t+lag
        w_t (tensor): weights at time t
        w_lag (tensor): weights at time t+lag
    '''
    #define lists
    x_t = []
    x_lag = []
    w_t = []
    w_lag = []
    #find maximum time idx
    idx_end = closest_idx(t,t[-1]-lag_time)
    #start_j = 0
    N = len(t)
    
    #loop over time array and find pairs which are far away by lag_time
    for i in range(idx_end):
        stop_condition = lag_time+t[i+1]
        n_j = 0
        
        for j in range(i,N):
            if ( t[j] < stop_condition ) and (t[j+1]>t[i]+lag_time):
                x_t.append(x[i])
                x_lag.append(x[j])
                deltaTau=min(t[i+1]+lag_time,t[j+1]) - max(t[i]+lag_time,t[j])
                w_lag.append(deltaTau)
                #if n_j == 0: #assign j as the starting point for the next loop
                #    start_j = j
                n_j +=1
            elif t[j] > stop_condition:
                break
        for k in range(n_j):
            w_t.append((t[i+1]-t[i])/float(n_j))

    x_t = torch.stack(x_t) if type(x) == torch.Tensor else torch.Tensor(x_t)
    x_lag = torch.stack(x_lag) if type(x) == torch.Tensor else torch.Tensor(x_lag)
    
    w_t = torch.Tensor(w_t)
    w_lag = torch.Tensor(w_lag)

    return x_t,x_lag,w_t,w_lag

def create_time_lagged_dataset(X, t = None, lag_time = 1, logweights = None, tprime = None, interval = None):
    """
    Create a dataset of time-lagged configurations. If a set of (log)weights is given the search is performed in the accelerated time.

    Parameters
    ----------
    X : array-like
        input descriptors
    t : array-like, optional
        time series, by default np.arange(len(X))
    lag_time: float, optional
        lag between configurations, by default = 10        
    logweights : array-like,optional
        logweights to evaluate rescaled time as dt' = dt*exp(logweights)
    tprime : array-like,optional
        rescaled time estimated from the simulation. If not given 'tprime_evaluation(t,logweights)' is used instead
    interval : list or np.array or tuple, optional
        Range for slicing the returned dataset. Useful to work with batches of same sizes.
        Recall that with different lag_times one obtains different datasets, with different lengths 
    """

    # check if dataframe
    if type(X) == pd.core.frame.DataFrame:
        X = X.values
    if type(t) == pd.core.frame.DataFrame:
        t = t.values

    # define time if not given
    if t is None:
        t = torch.arange(0,len(X))
    else:
        if len(t) != len(X):
            raise ValueError(f'The length of t ({len(t)}) is different from the one of X ({len(X)}) ')

    #define tprime if not given
    if tprime is None:
        tprime = tprime_evaluation(t, logweights)

    # find pairs of configurations separated by lag_time
    data = find_time_lagged_configurations(X, tprime,lag_time=lag_time)

    if interval is not None:
        # convert to a list
        data = list(data)
        # assert dimension of interval
        assert len(interval) == 2
        # modifies the content of data by slicing
        for i in range(len(data)):
            data[i] = data[i][interval[0]:interval[1]]

    #return data
    return torch.utils.data.TensorDataset(*data)

def test_create_time_lagged_dataset():
    in_features = 2
    n_points = 100
    X = torch.rand(n_points,in_features)*100

    dataset = create_time_lagged_dataset(X)
    print(len(dataset))

    t = np.arange(n_points)
    dataset = create_time_lagged_dataset(X,t,lag_time=10)
    print(len(dataset))

    logweights = np.random.rand(n_points)
    dataset =  create_time_lagged_dataset(X,t,logweights=logweights)
    print(len(dataset))  

if __name__ == "__main__":
    test_create_time_lagged_dataset()