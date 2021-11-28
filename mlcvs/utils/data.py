"""Datasets."""

import torch 
import numpy as np

__all__ = ["LabeledDataset","create_time_lagged_dataset","FastTensorDataLoader"]

from torch.utils.data import Dataset,Subset
from bisect import bisect_left

class LabeledDataset(Dataset):
    """
    Dataset with labels.
    """

    def __init__(self, colvar, labels):
        """
        Create dataset from colvar and labels.

        Parameters
        ----------
        colvar : array-like
            input data 
        labels : array-like
            classes labels
        """

        self.colvar = colvar
        self.labels = labels

    def __len__(self):
        return len(self.colvar)

    def __getitem__(self, idx):
        x = (self.colvar[idx], self.labels[idx])
        return x

def closest_idx_torch(array, value):
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

def find_time_lagged_configurations(x,t,lag):
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
    idx_end = closest_idx_torch(t,t[-1]-lag)
    #start_j = 0
    
    #loop over time array and find pairs which are far away by lag
    for i in range(idx_end):
        stop_condition = lag+t[i+1]
        n_j = 0
        
        for j in range(i,len(t)):
            if ( t[j] < stop_condition ) and (t[j+1]>t[i]+lag):
                x_t.append(x[i])
                x_lag.append(x[j])
                deltaTau=min(t[i+1]+lag,t[j+1]) - max(t[i]+lag,t[j])
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

def create_time_lagged_dataset(X, t = None, lag_time = 10, logweights = None):
    """
    Create dataset of time-lagged configurations, with optional (log)weights to reweight a biased trajectory. If the weights are given, the calculation of the time-lagged correlation functions is done in the rescaled time t'.  

    TODO DOC
    """

    # define time if not given
    if t is None:
        t = np.arange(0,len(X))

    # rescale time with log-weights
    if logweights is not None:
        # compute time increment in simulation time t
        dt = np.round(t[1]-t[0],3)
        # compute instantaneus time increment in rescaled time t'
        d_tprime = np.copy(np.exp(logweights)*dt)
        #calculate cumulative time tau
        tprime = np.cumsum(d_tprime)
    else:
        tprime = t

    # find pairs of configurations separated by lag_time
    data = find_time_lagged_configurations(X, tprime,lag=lag_time)

    #return data
    return torch.utils.data.TensorDataset(*data)

class FastTensorDataLoader:
    """
    A DataLoader-like object for a set of tensors.
    
    It is much faster than TensorDataset + DataLoader because dataloader grabs individual indices of the dataset and calls cat (slow).

    Notes
    =====

    Adapted from https://discuss.pytorch.org/t/dataloader-much-slower-than-manual-batching/27014/6. 

    """
    def __init__(self, tensors, batch_size=0, shuffle=False):
        """Initialize a FastTensorDataLoader.

        Parameters
        ----------
        tensors : list of tensors or torch.Dataset or torch.Subset object containing a tensors object
            tensors to store. Must have the same length @ dim 0.
        batch_size : int, optional
            batch size, by default 0 (==single batch)
        shuffle : bool, optional
            if True, shuffle the data *in-place* whenever an
            iterator is created out of this object, by default False

        Returns
        -------
        FastTensorDataLoader
            dataloader-like object

        """

        # check input type
        if type(tensors) == Dataset:
            tensors = [ tensors.tensors[i] for i in range(len(tensors.tensors)) ]
        elif type(tensors) == Subset:
            tensors = [ tensors.dataset.tensors[i][tensors.indices] for i in range(len(tensors.dataset.tensors)) ]

        assert all(t.shape[0] == tensors[0].shape[0] for t in tensors)
        self.tensors = tensors

        self.dataset_len = self.tensors[0].shape[0]
        self.batch_size = batch_size if batch_size > 0 else self.dataset_len
        self.shuffle = shuffle

        # Calculate # batches
        n_batches, remainder = divmod(self.dataset_len, self.batch_size)
        if remainder > 0:
            n_batches += 1
        self.n_batches = n_batches

    def __iter__(self):
        if self.shuffle:
            self.indices = torch.randperm(self.dataset_len)
        else:
            self.indices = None
        self.i = 0
        return self

    def __next__(self):
        if self.i >= self.dataset_len:
            raise StopIteration
        if self.indices is not None:
            indices = self.indices[self.i:self.i+self.batch_size]
            batch = tuple(torch.index_select(t, 0, indices) for t in self.tensors)
        else:
            batch = tuple(t[self.i:self.i+self.batch_size] for t in self.tensors)
        self.i += self.batch_size
        return batch

    def __len__(self):
        return self.n_batches


