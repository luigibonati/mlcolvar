import torch
import numpy as np
from typing import Tuple

from mlcolvar.graph import data as gdata
from mlcolvar.utils.timelagged import (
    tprime_evaluation,
    find_timelagged_configurations
)

try:
    import pandas as pd

    PANDAS_IS_INSTALLED = True
except ImportError:
    PANDAS_IS_INSTALLED = False
try:
    from tqdm import tqdm

    TQDM_IS_INSTALLED = True
except ImportError:
    TQDM_IS_INSTALLED = False

"""
Build time-lagged dataset for GNNs.
"""

__all__ = ['create_timelagged_datasets']


def create_timelagged_datasets(
    dataset: gdata.GraphDataSet,
    t: torch.Tensor = None,
    lag_time: float = 1,
    reweight_mode: str = None,
    logweights: torch.Tensor = None,
    tprime: torch.Tensor = None,
    interval: list = None,
    progress_bar: bool = False,
) -> Tuple[gdata.GraphDataSet, gdata.GraphDataSet]:
    """
    Create two GraphDataSets of time-lagged configurations.

    In case of biased simulations the reweight can be performed in two
    different ways (``reweight_mode``):

    1. ``rescale_time`` : the search for time-lagged pairs is performed in the
        accelerated time (dt' = dt*exp(logweights)), as described in [1]_ .
    2. ``weights_t`` : the weight of each pair of configurations (t,t+lag_time)
        depends only on time t (logweights(t)), as done in [2]_ , [3]_ .

    If reweighting is None and tprime is given the `rescale_time` mode is used.
    If instead only the logweights are specified the user needs to choose the
    reweighting mode.

    References
    ----------
    .. [1] Y. I. Yang and M. Parrinello, "Refining collective coordinates and
        improving free energy representation in variational enhanced sampling."
        JCTC 14, 2889–2894 (2018).
    .. [2] J. McCarty and M. Parrinello, "A variational conformational dynamics
        approach to the selection of collective variables in meta-dynamics."
        JCP 147, 204109 (2017).
    .. [3] H. Wu, et al. "Variational Koopman models: Slow collective variables
        and molecular kinetics from short off-equilibrium simulations."
        JCP 146.15 (2017).

    Parameters
    ----------
    dataset : mlcolvar.graph.data.GraphDataSet
        The reference dataset.
    t : array-like, optional
        Time series, by default np.arange(len(X))
    reweight_mode: str, optional
        How to do the reweighting, see documentation, by default none
    lag_time: float, optional
        Lag between configurations, by default = 10
    logweights : array-like,optional
        Logweight of each configuration (typically beta*bias)
    tprime : array-like,optional
        Rescaled time estimated from the simulation. If not given and
        `reweighting_mode`=`rescale_time` then
        `tprime_evaluation(t,logweights)` is used
    interval : list or np.array or tuple, optional
        Range for slicing the returned dataset. Useful to work with batches of
        same sizes. Recall that with different lag_times one obtains different
        datasets, with different lengths
    progress_bar: bool
        If display progress bar with tqdm.

    Returns
    -------
    datasets: Tuple[GraphDataSet, GraphDataSet]
        Datasets at time t and t+lag.
    """

    if PANDAS_IS_INSTALLED and isinstance(t, pd.core.frame.DataFrame):
        t = t.values

    # check reweigthing mode if logweights are given:
    # 1) if rescaled time tprime is given
    if tprime is not None:
        if reweight_mode is None:
            reweight_mode = 'rescale_time'
        elif reweight_mode != 'rescale_time':
            raise ValueError(
                'The `reweighting_mode` needs to be equal to `rescale_time`, '
                + 'and not {reweight_mode} if the rescale time `tprime` is '
                + 'given.'
            )
    # 2) if logweights are given
    elif logweights is not None:
        if reweight_mode is None:
            reweight_mode = 'rescale_time'
            # TODO output warning or error if mode not specified?
            # warnings.warn(
            #   '`reweight_mode` not specified, setting it to `rescale_time`.'
            # )

    # define time if not given
    if t is None:
        t = torch.arange(0, len(dataset))
    else:
        if len(t) != len(dataset):
            message = (
                'The length of t ({:d}) is different from the one of '
                + 'dataset ({:d}).'
            )
            raise ValueError(message.format(len(t), len(dataset)))

    # define tprime if not given:
    if reweight_mode == 'rescale_time':
        if tprime is None:
            tprime = tprime_evaluation(t, logweights)
    else:
        tprime = t

    # find pairs of configurations separated by lag_time
    index = torch.arange(len(dataset), dtype=torch.long)
    x_t, x_lag, w_t, w_lag = find_timelagged_configurations(
        index,
        tprime,
        lag_time=lag_time,
        logweights=logweights if reweight_mode == 'weights_t' else None,
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
            data[i] = data[i][interval[0]:interval[1]]
        x_t, x_lag, w_t, w_lag = data

    assert len(x_t) == len(w_t)
    assert len(x_t) == len(x_lag)
    assert len(w_t) == len(w_lag)

    # assign weights
    dataset_t = dataset[x_t.numpy().tolist()]
    dataset_lag = dataset[x_lag.numpy().tolist()]

    for i in range(len(x_t)):
        dataset_t[i]['weight'] = w_t[i]
        dataset_lag[i]['weight'] = w_lag[i]

    return (dataset_t, dataset_lag)


def test_timelagged() -> None:
    from mlcolvar.utils.timelagged import create_timelagged_dataset as _ctd

    numbers = [8, 1, 1]
    positions = np.array(
        [[0.0, 0.0, 0.0], [0.07, 0.07, 0.0], [0.07, -0.07, 0.0]], dtype=float
    )
    cell = np.identity(3, dtype=float) * 0.2
    graph_labels = np.array([[1]])
    node_labels = np.array([[0], [1], [1]])
    z_table = gdata.atomic.AtomicNumberTable.from_zs(numbers)

    config = gdata.atomic.Configuration(
        atomic_numbers=numbers,
        positions=positions,
        cell=cell,
        pbc=[True] * 3,
        node_labels=node_labels,
        graph_labels=graph_labels,
    )
    dataset = gdata.create_dataset_from_configurations(
        [config] * 10, z_table, 0.1, show_progress=False
    )
    for i in range(len(dataset)):
        dataset[i]['graph_labels'] *= i

    datasets = create_timelagged_datasets(dataset)
    data_reference = _ctd(torch.arange(10))

    assert len(datasets[0]) == len(data_reference['data'])
    assert len(datasets[1]) == len(data_reference['data_lag'])

    for i in range(len(datasets[0])):
        d_0 = datasets[0]
        d_ref = data_reference['data']
        w_ref = data_reference['weights']
        assert d_0[i]['weight'] == w_ref[i]
        assert d_0[i]['graph_labels'][0, 0] == d_ref[i]
    for i in range(len(datasets[1])):
        d_1 = datasets[1]
        d_ref = data_reference['data_lag']
        w_ref = data_reference['weights_lag']
        assert d_1[i]['weight'] == w_ref[i]
        assert d_1[i]['graph_labels'][0, 0] == d_ref[i]

    datasets = create_timelagged_datasets(dataset, lag_time=2)
    data_reference = _ctd(torch.arange(10), lag_time=2)

    assert len(datasets[0]) == len(data_reference['data'])
    assert len(datasets[1]) == len(data_reference['data_lag'])

    for i in range(len(datasets[0])):
        d_0 = datasets[0]
        d_ref = data_reference['data']
        w_ref = data_reference['weights']
        assert d_0[i]['weight'] == w_ref[i]
        assert d_0[i]['graph_labels'][0, 0] == d_ref[i]
    for i in range(len(datasets[1])):
        d_1 = datasets[1]
        d_ref = data_reference['data_lag']
        w_ref = data_reference['weights_lag']
        assert d_1[i]['weight'] == w_ref[i]
        assert d_1[i]['graph_labels'][0, 0] == d_ref[i]


if __name__ == '__main__':
    test_timelagged()
