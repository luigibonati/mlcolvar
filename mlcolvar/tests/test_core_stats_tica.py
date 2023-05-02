import pytest
import torch

from mlcolvar.utils.io import load_dataframe
from mlcolvar.utils.timelagged import create_timelagged_dataset
from mlcolvar.core.stats import TICA

@pytest.mark.parametrize("algorithm", ['least_squares','reduced_rank'])
def test_tica(algorithm):
    # fake data
    in_features = 2
    X = torch.rand(100,in_features)*100
    x_t = X[:-1]
    x_lag = X[1:]
    w_t = torch.rand(len(x_t))
    w_lag = w_t
    
    # define
    tica = TICA(in_features,out_features=1)
    # compute
    tica.compute(data=[x_t,x_lag],
                 weights=[w_t,w_lag], 
                 algorithm=algorithm,
                 save_params=True)
    # project
    s = tica(X)

    print(X.shape,'-->',s.shape)
    print('eigvals',tica.evals)
    print('timescales', tica.timescales(lag=10))


@pytest.mark.parametrize("algorithm", ['least_squares','reduced_rank'])
def test_tica_from_dataset(algorithm):
    # load data
    df = load_dataframe('mlcolvar/tests/data/state_A.dat')
    X = df.filter(regex='n').values
    X = torch.Tensor(X)
    dataset = create_timelagged_dataset(X)
    
    # define
    tica = TICA(in_features=2,out_features=1)
    # compute
    tica.compute(data=[dataset['data'],dataset['data_lag']],
                 weights=[dataset['weights'],dataset['weights_lag']], 
                 algorithm=algorithm,
                 save_params=True)
    # project
    s = tica(X)

    print(X.shape,'-->',s.shape)
    print('eigvals',tica.evals)
    print('timescales', tica.timescales(lag=10))
