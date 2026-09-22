import torch

from mlcolvar.core.estimators.tica import TICA
from mlcolvar.core.estimators.utils import correlation_matrix, cholesky_eigh


def test_tica():
    in_features = 2

    X = torch.rand(100, in_features) * 100
    x_t = X[:-1]
    x_lag = X[1:]

    w_t = torch.rand(len(x_t))
    w_lag = w_t

    # direct way, compute TICA
    tica = TICA(in_features, out_features=2)
    print(tica)

    tica.compute(
        [x_t, x_lag],
        [w_t, w_lag],
        save_params=True,
    )

    s = tica(X)

    print(X.shape, "-->", s.shape)
    print("eigvals", tica.evals)
    print("timescales", tica.timescales(lag=10))

    # step by step
    tica = TICA(in_features)

    C_0 = correlation_matrix(x_t, x_t)
    C_lag = correlation_matrix(x_t, x_lag)

    print(C_0.shape, C_lag.shape)

    evals, evecs = cholesky_eigh(C_lag, C_0)

    print(evals.shape, evecs.shape)

    print(">> batch")
    s = tica(X)
    print(X.shape, "-->", s.shape)

    print(">> single")
    X2 = X[0]
    s2 = tica(X2)
    print(X2.shape, "-->", s2.shape)