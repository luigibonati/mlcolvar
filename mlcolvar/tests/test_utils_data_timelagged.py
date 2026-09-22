import numpy as np
import torch

from mlcolvar.data import DictDataset
from mlcolvar.data.graph.utils import create_test_graph_input
from mlcolvar.utils.timelagged import create_timelagged_dataset


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
    dataset = create_test_graph_input('dataset')
    print(dataset['data_list'][0])
    lagged_dataset = create_timelagged_dataset(dataset, logweights=torch.randn(len(dataset)))
    print(lagged_dataset['data_list'][0])
    print(dataset['data_list'][0])

    print(len(dataset))
    
    # unbiased multi-walker case
    lag_time = 5
    walker = np.array([0] * (n_points // 2) + [1] * (n_points // 2))
    dataset = create_timelagged_dataset(
        X, t, lag_time=lag_time, walker=walker
    )
    assert len(dataset) == n_points - 2 * lag_time