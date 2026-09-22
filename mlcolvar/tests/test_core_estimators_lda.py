import torch

from mlcolvar.core.estimators.lda import LDA


def test_lda():
    in_features = 2
    n_states = 2

    torch.manual_seed(42)
    X = torch.rand(100, in_features) * 100
    y = torch.randint(n_states, (100, 1)).squeeze(1)

    # standard
    lda = LDA(in_features, n_states)
    print(lda)

    S_b, S_w = lda.compute_scatter_matrices(X, y)
    print(S_w, S_b)

    evals, evecs = lda.compute(X, y, True)
    print(lda.S_w.shape, lda.S_b.shape)
    print(evals.shape, evecs.shape)

    s = lda(X)
    print(s.shape)

    assert (s.ndim == 2) and (s.shape[1] == n_states - 1)

    # harmonic variant
    hlda = LDA(in_features, n_states, mode="harmonic")
    print(hlda)

    hlda.compute(X, y)
    s = hlda(X)

    assert (s.ndim == 2) and (s.shape[1] == n_states - 1)
