import torch
import numpy as np
from bisect import bisect_left
from mlcolvar.data import DictDataset
import warnings
from typing import Union
import copy

# optional packages
# pandas
try:
    import pandas as pd

    PANDAS_IS_INSTALLED = True
except ImportError:
    PANDAS_IS_INSTALLED = False
# tqdm (progress bar)
try:
    from tqdm import tqdm

    TQDM_IS_INSTALLED = True
except ImportError:
    TQDM_IS_INSTALLED = False

__all__ = ["find_timelagged_configurations", "create_timelagged_dataset"]


def closest_idx(array, value):
    """
    Find index of the element of 'array' which is closest to 'value'.
    The array is first converted to a np.array in case of a tensor.
    Note: it does always round to the lowest one.

    Parameters:
        array (tensor/np.array)
        value (float)

    Returns:
        pos (int): index of the closest value in array
    """
    if type(array) is np.ndarray:
        pos = bisect_left(array, value)
    else:
        pos = bisect_left(array.numpy(), value)
    if pos == 0:
        return 0
    elif pos == len(array):
        return -1
    else:
        return pos - 1


# evaluation of tprime from simulation time and logweights
def tprime_evaluation(t, logweights=None):
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
        dt = np.round(t[1] - t[0], 5)
        # sanitize logweights
        logweights = torch.Tensor(logweights)
        # when the bias is not deposited the value of bias potential is minimum
        logweights -= torch.max(logweights)
        # note: exp(logweights/lognorm) != exp(logweights)/norm, where norm is sum_i beta V_i
        """ possibilities:
            1) logweights /= torch.min(logweights) -> logweights belong to [0,1] 
            2) pass beta as an argument, then logweights *= beta
            3) tprime = dt * torch.cumsum( torch.exp( torch.logsumexp(logweights,0) ) ,0)
            4) tprime = dt *torch.exp ( torch.log (torch.cumsum (torch.exp(logweights) ) ) )
        """
        lognorm = torch.logsumexp(logweights, 0)
        logweights /= lognorm
        # compute instantaneus time increment in rescaled time t'
        d_tprime = torch.exp(logweights) * dt
        # calculate cumulative time t'
        tprime = torch.cumsum(d_tprime, 0)
    else:
        tprime = t

    return tprime


def find_timelagged_configurations(
    x: torch.Tensor,
    t: torch.Tensor,
    lag_time: float,
    logweights: torch.Tensor = None,
    progress_bar: bool = True,
):
    """Searches for all the pairs which are distant 'lag' in time, and returns the weights associated.
    If logweights are provided they will be returned both for x_t and x_t+lag (used only for `reweight_mode=weights_t` of create_time_lagged_dataset).

    Parameters
    ----------
    x : torch.Tensor
        array whose columns are the descriptors and rows the time evolution
    t : torch.Tensor
        array with the simulation time
    lag_time : float
        lag-time
    logweights : torch.Tensor, optional
        logweights to be returned
    progress_bar : bool, optional
        display progress bar with tqdm (if installed), by default True

    Returns
    -------
    x_t: torch.Tensor
        descriptors at time t
    x_lag: torch.Tensor
        descriptors at time t+lag
    w_t: torch.Tensor
        weights at time t
    w_lag: torch.Tensor
        weights at time t+lag
    """

    # define lists
    x_t = []
    x_lag = []
    w_t = []
    w_lag = []
    # find maximum time idx
    idx_end = closest_idx(t, t[-1] - lag_time)
    # start_j = 0
    N = len(t)

    def progress(iter, progress_bar=progress_bar):
        if progress_bar and TQDM_IS_INSTALLED:
            return tqdm(iter)
        else:
            warnings.warn(
                "Monitoring the progress for the search of time-lagged configurations with a progress_bar requires `tqdm`."
            )
            return iter

    # sanitize logweights if given
    calculate_weights = True
    if logweights is not None:
        calculate_weights = False
        if len(logweights) != len(x):
            raise ValueError(
                f"Length of logweights ({len(logweights)}) is different from length of data ({len(x)})."
            )
        logweights = torch.Tensor(logweights)
        weights = torch.exp(logweights)

    # loop over time array and find pairs which are far away by lag_time
    for i in progress(range(idx_end)):
        stop_condition = lag_time + t[i + 1]
        n_j = 0

        for j in range(i, N):
            if (t[j] < stop_condition) and (t[j + 1] > t[i] + lag_time):
                x_t.append(x[i])
                x_lag.append(x[j])
                deltaTau = min(t[i + 1] + lag_time, t[j + 1]) - max(
                    t[i] + lag_time, t[j]
                )

                if calculate_weights:
                    w_lag.append(deltaTau)
                else:
                    w_lag.append(weights[i])
                # if n_j == 0: #assign j as the starting point for the next loop
                #    start_j = j
                n_j += 1
            elif t[j] > stop_condition:
                break
        for k in range(n_j):
            if calculate_weights:
                w_t.append((t[i + 1] - t[i]) / float(n_j))
            else:
                if n_j > 1:
                    print(n_j)
                w_t.append(weights[i] / float(n_j))

    x_t = torch.stack(x_t) if type(x) == torch.Tensor else torch.Tensor(x_t)
    x_lag = torch.stack(x_lag) if type(x) == torch.Tensor else torch.Tensor(x_lag)

    w_t = torch.Tensor(w_t)
    w_lag = torch.Tensor(w_lag)

    return x_t, x_lag, w_t, w_lag


def create_timelagged_dataset(
    X: Union[torch.Tensor, np.ndarray, DictDataset],
    t: torch.Tensor = None,
    lag_time: float = 1,
    reweight_mode: str = None,
    logweights: torch.Tensor = None,
    tprime: torch.Tensor = None,
    interval: list = None,
    progress_bar: bool = False,
):
    """
    Create a DictDataset of time-lagged configurations.

    In case of biased simulations the reweight can be performed in two different ways (``reweight_mode``):

    1. ``rescale_time`` : the search for time-lagged pairs is performed in the accelerated time (dt' = dt*exp(logweights)), as described in [1]_ .
    2. ``weights_t`` : the weight of each pair of configurations (t,t+lag_time) depends only on time t (logweights(t)), as done in [2]_ , [3]_ .

    If reweighting is None and tprime is given the `rescale_time` mode is used. If instead only the logweights are specified the user needs to choose the reweighting mode.

    References
    ----------
    .. [1] Y. I. Yang and M. Parrinello, “Refining collective coordinates and improving free energy
        representation in variational enhanced sampling,” JCTC 14, 2889–2894 (2018).
    .. [2] J. McCarty and M. Parrinello, "A variational conformational dynamics approach to the selection
        of collective variables in meta- dynamics,” JCP 147, 204109 (2017).
    .. [3] H. Wu, et al. "Variational Koopman models: Slow collective variables and molecular kinetics
        from short off-equilibrium simulations." JCP 146.15 (2017).

    Parameters
    ----------
    X : torch.Tensor or np.ndarray or DictDataset
        Input data, graph data can only be provided as DictDataset
    t : array-like, optional
        time series, by default np.arange(len(X))
    reweight_mode: str, optional
        how to do the reweighting, see documentation, by default none
    lag_time: float, optional
        lag between configurations, by default = 10
    logweights : array-like,optional
        logweight of each configuration (typically beta*bias)
    tprime : array-like,optional
        rescaled time estimated from the simulation. If not given and `reweighting_mode`=`rescale_time` then `tprime_evaluation(t,logweights)` is used
    interval : list or np.array or tuple, optional
        Range for slicing the returned dataset. Useful to work with batches of same sizes. Recall that with different lag_times one obtains different datasets, with different lengths
    progress_bar: bool
        Display progress bar with tqdm

    Returns
    -------
    dataset: DictDataset
        Dataset with keys 'data', 'data_lag' (data at time t and t+lag), 'weights', 'weights_lag' (weights at time t and t+lag).

    """

    if PANDAS_IS_INSTALLED:
        # check if dataframe
        if type(X) == pd.core.frame.DataFrame:
            X = X.values
        if type(t) == pd.core.frame.DataFrame:
            t = t.values

    # check reweigthing mode if logweights are given:
    # 1) if rescaled time tprime is given
    if tprime is not None:
        if reweight_mode is None:
            reweight_mode = "rescale_time"
        elif reweight_mode != "rescale_time":
            raise ValueError(
                "The `reweighting_mode` needs to be equal to `rescale_time`, and not {reweight_mode} if the rescale time `tprime` is given."
            )
    # 2) if logweights are given
    elif logweights is not None:
        if reweight_mode is None:
            reweight_mode = "rescale_time"
            # TODO output warning or error if mode not specified?
            # warnings.warn('`reweight_mode` not specified, setting it to `rescale_time`.')

    # define time if not given
    if t is None:
        t = torch.arange(0, len(X))
    else:
        if len(t) != len(X):
            raise ValueError(
                f"The length of t ({len(t)}) is different from the one of X ({len(X)}) "
            )

    # define tprime if not given:
    if reweight_mode == "rescale_time":
        if tprime is None:
            tprime = tprime_evaluation(t, logweights)
    else:
        tprime = t

    # find pairs of configurations separated by lag_time
    if isinstance(X, torch.Tensor) or isinstance(X, np.ndarray):
        x_t, x_lag, w_t, w_lag = find_timelagged_configurations(
            X,
            tprime,
            lag_time=lag_time,
            logweights=logweights if reweight_mode == "weights_t" else None,
            progress_bar=progress_bar,
        )
    elif isinstance(X, DictDataset):
        index = torch.arange(len(X), dtype=torch.long)
        x_t, x_lag, w_t, w_lag = find_timelagged_configurations(
            index,
            tprime,
            lag_time=lag_time,
            logweights=logweights if reweight_mode == "weights_t" else None,
            progress_bar=progress_bar,
        )

    # return only a slice of the data (N. Pedrani)
    if interval is not None:
        # convert to a list
        data = list(x_t, x_lag, w_t, w_lag)
        # assert dimension of interval
        assert len(interval) == 2
        # modifies the content of data by slicing
        for i in range(len(data)):
            data[i] = data[i][interval[0] : interval[1]]
        x_t, x_lag, w_t, w_lag = data

    if isinstance(X, torch.Tensor) or isinstance(X, np.ndarray):
        dataset = DictDataset({"data": x_t, 
                               "data_lag": x_lag, 
                               "weights": w_t, 
                               "weights_lag": w_lag},
                               data_type='descriptors')
        return dataset 
    
    elif isinstance(X, DictDataset):
        if X.metadata["data_type"] == "descriptors":
            dataset = DictDataset({"data": X['data'][x_t], 
                               "data_lag": X['data'][x_lag], 
                               "weights": w_t, 
                               "weights_lag": w_lag},
                               data_type='descriptors')
            
        elif X.metadata["data_type"] == "graphs":
            # we use deepcopy to avoid editing the original dataset
            dataset = DictDataset(dictionary={"data_list" : copy.deepcopy(X[x_t.numpy().tolist()]["data_list"]),
                                            "data_list_lag" : copy.deepcopy(X[x_lag.numpy().tolist()]["data_list"])},
                                    metadata={"z_table" : X.metadata["z_table"],
                                            "cutoff" : X.metadata["cutoff"]},
                                    data_type="graphs")
            # update weights
            for i in range(len(dataset)):
                dataset['data_list'][i]['weight'] = w_t[i]
                dataset['data_list_lag'][i]['weight'] = w_lag[i]
            
        return dataset


def test_create_timelagged_dataset():
    in_features = 2
    n_points = 20
    X = torch.rand(n_points, in_features) * 100
    dataset = DictDataset(data=X, data_type='descriptors')


    # unbiased case
    t = np.arange(n_points)
    lagged_dataset_1 = create_timelagged_dataset(X, t, lag_time=10)
    print(len(lagged_dataset_1))
    lagged_dataset_2 = create_timelagged_dataset(dataset, t, lag_time=10)
    print(len(lagged_dataset_2))
    assert(torch.allclose(lagged_dataset_1['data'], lagged_dataset_2['data']))
    assert(torch.allclose(lagged_dataset_1['data_lag'], lagged_dataset_2['data_lag']))
    assert(torch.allclose(lagged_dataset_1['weights'], lagged_dataset_2['weights']))


    # reweight mode rescale_time (default)
    logweights = np.random.rand(n_points)
    lagged_dataset_1 = create_timelagged_dataset(X, t, logweights=logweights)
    print(len(lagged_dataset_1))
    lagged_dataset_2 = create_timelagged_dataset(dataset, t, logweights=logweights)
    print(len(lagged_dataset_2))
    assert(torch.allclose(lagged_dataset_1['data'], lagged_dataset_2['data']))
    assert(torch.allclose(lagged_dataset_1['data_lag'], lagged_dataset_2['data_lag']))
    assert(torch.allclose(lagged_dataset_1['weights'], lagged_dataset_2['weights']))


    # reweight mode weights_t
    logweights = np.random.rand(n_points)
    lagged_dataset_1 = create_timelagged_dataset(
        X, t, logweights=logweights, reweight_mode="weights_t"
    )
    print(len(lagged_dataset_1))
    lagged_dataset_2 = create_timelagged_dataset(
        dataset, t, logweights=logweights, reweight_mode="weights_t"
    )
    print(len(lagged_dataset_2))
    assert(torch.allclose(lagged_dataset_1['data'], lagged_dataset_2['data']))
    assert(torch.allclose(lagged_dataset_1['data_lag'], lagged_dataset_2['data_lag']))
    assert(torch.allclose(lagged_dataset_1['weights'], lagged_dataset_2['weights']))



    # graph data
    from mlcolvar.data.graph.utils import create_test_graph_input
    dataset = create_test_graph_input('dataset')
    print(dataset['data_list'][0])
    lagged_dataset = create_timelagged_dataset(dataset, logweights=torch.randn(len(dataset)))
    print(lagged_dataset['data_list'][0])
    print(dataset['data_list'][0])

    print(len(dataset))
    


if __name__ == "__main__":
    test_create_timelagged_dataset()
